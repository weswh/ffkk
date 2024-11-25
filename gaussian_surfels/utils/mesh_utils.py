#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#
import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh
import cv2
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point
import json
import threading
import time
import concurrent.futures
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import torchvision.transforms.functional as TF
from concurrent.futures import ThreadPoolExecutor
from numba import jit, prange

def post_process_mesh(mesh, image_path,cluster_to_keep=50):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    vertices = np.asarray(mesh_0.vertices)
    # rotation_matrix = np.array([
    #     [1,0,0],
    #     [0,0,1],
    #     [0,-1,0]
    # ])
    # rotated_vertices=np.dot(vertices,rotation_matrix.T)
    # mesh_0.vertices =o3d.utility.Vector3dVector(rotated_vertices)

    # 调整模型位置，y轴朝上，脚尖朝向z轴正方向
    # translation_vector=np.array([0,0.7,0])
    # translated_vertices = vertices + translation_vector
    # mesh_0.vertices =o3d.utility.Vector3dVector(translated_vertices)
    # vertices = np.asarray(mesh_0.vertices)
    # rotation_matrix = np.array([
    #     [0,0,-1],
    #     [0,1,0],
    #     [1,0,0]
    # ])
    # rotated_vertices=np.dot(vertices,rotation_matrix.T)
    # mesh_0.vertices =o3d.utility.Vector3dVector(rotated_vertices)

    # # 修剪模型
    # vertices = np.asarray(mesh_0.vertices)
    # vertices_to_keep = vertices[:, 1] >= 1.5
    # print('vertices_to_keep',vertices_to_keep)
    # triangles_to_remove = np.any(vertices_to_keep[mesh_0.triangles], axis=1)
    # mesh_0.remove_triangles_by_mask(triangles_to_remove)
    # mesh_0.remove_unreferenced_vertices()
    # mesh_0.remove_degenerate_triangles()

    # vertices_to_keep2 = vertices[:, 1] <= 0
    # triangles_to_remove = np.any(vertices_to_keep2[mesh_0.triangles], axis=1)
    # mesh_0.remove_triangles_by_mask(triangles_to_remove)
    # mesh_0.remove_unreferenced_vertices()
    # mesh_0.remove_degenerate_triangles()

    #删掉冗余点
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 10000) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    json_path =os.path.join(image_path,'transforms.json')
    vertices = np.asarray(mesh_0.vertices)
    mesh_0=project(json_path, image_path, vertices, mesh_0)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    return mesh_0



@jit(nopython=True)
def inverse_transform_matrix(matrix):
    matrix[0:3, 1:3] *= -1
    return np.linalg.inv(matrix)

@jit(nopython=True)
def transform_vertices(vertices, matrix):
    # Extract rotation and translation parts of the matrix
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    
    # Apply transformation using broadcasting
    transformed = np.dot(vertices, rotation.T) + translation
    return transformed

@jit(nopython=True)
def process_image_points(image_points):
    x_coords, y_coords = image_points[:, 0], image_points[:, 1]
    return x_coords, y_coords

@jit(nopython=True)
def get_colors_at_points(image, x_coords, y_coords):
    colors_at_points = np.zeros((len(x_coords), 3), dtype=np.uint8)
    for i in range(len(x_coords)):
        colors_at_points[i] = image[y_coords[i], x_coords[i]]
    return colors_at_points

@jit(nopython=True)
def all_zero(arr):
    # Vectorized implementation to check if all elements along the last axis are zero
    return np.sum(arr, axis=1) == 0

@jit(nopython=True)
def get_valid_indices(colors_at_points):
    # return np.all(colors_at_points == 0)
    return all_zero(colors_at_points)


def process_frame(frame_data, image_path, vertices, mesh_0 ,  triangles_to_remove_list):
    mask_path = os.path.join(image_path, 'u_masks/select/')
    mask_path = mask_path + frame_data['file_path'].split('/')[-1]
    fl_x = frame_data['fl_x']
    fl_y = frame_data['fl_y']
    cx = frame_data['cx']
    cy = frame_data['cy']
    transform_matrix = np.array(frame_data['transform_matrix'], dtype=np.float64)
    matrix = inverse_transform_matrix(transform_matrix)
    image = cv2.imread(mask_path)
    camera_matrix = np.array([[fl_x, 0, cx],
                              [0, fl_y, cy],
                              [0, 0, 1]], dtype=np.float64)
    # transformed_vertices = transform_vertices(vertices, matrix)
    #投影到掩模图像
    transformed_vertices = np.matmul(vertices, matrix[:3, :3].T) + matrix[:3, 3]
    image_points, _ = cv2.projectPoints(transformed_vertices, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, None)
    image_points = np.int32(image_points).reshape(-1, 2)
    try:
        x_coords, y_coords= process_image_points(image_points)
        colors_at_points = image[y_coords, x_coords]
        valid_indices = get_valid_indices(colors_at_points)
        # print('valid_indices',valid_indices)
        if np.any(valid_indices):
            triangles_to_remove = np.any(valid_indices[mesh_0.triangles], axis=1)
            # print('triangles_to_remove',triangles_to_remove)
            triangles_to_remove_list.append(triangles_to_remove)
            if np.any(triangles_to_remove):
               print('There are black triangle in the image.')
        else:
            print('No black points found in the image.')
    except IndexError:
        pass

def project(json_path, image_path, vertices, mesh_0):
    start_time = time.time()
    with open(json_path, 'r') as f:
        data = json.load(f)

    # List to collect all triangles to remove
    triangles_to_remove_list = []

    # Create threads for each frame
    threads = []
    for frame_data in data['frames']:
        thread = threading.Thread(target=process_frame, args=(frame_data, image_path, vertices, mesh_0, triangles_to_remove_list))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    triangles_to_remove = [False] * len(mesh_0.triangles)

    # 初始化一个计数器数组，用于统计每个三角形的位置有多少个 True
    true_count = [0] * len(mesh_0.triangles)

    # 遍历每个掩码，并统计每个位置为 True 的次数
    for mask in triangles_to_remove_list:
        true_count = [true_count[i] + 1 if mask[i] else true_count[i] for i in range(len(true_count))]

    # 如果某个位置的 True 数量 >= 5，则在最终掩码中标记为 True
    triangles_to_remove = [count >= 3 for count in true_count]

    # 一次性删除所有三角形
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time} seconds")
    return mesh_0




def to_cam_open3d(viewpoint_stack):
    # camera_traj = []
    # for i, viewpoint_cam in enumerate(viewpoint_stack):
        viewpoint_cam=viewpoint_stack
        intrinsic=o3d.camera.PinholeCameraIntrinsic(width=viewpoint_cam.image_width, 
                    height=viewpoint_cam.image_height, 
                    # cx = viewpoint_cam.image_width/2,
                    # cy = viewpoint_cam.image_height/2,
                    cx = viewpoint_cam.prcppoint[0]* viewpoint_cam.image_width ,
                    cy = viewpoint_cam.prcppoint[1]* viewpoint_cam.image_height,
                    #print(cx,cy)
                    fx = viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx / 2.)),
                    fy = viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy / 2.)))
        #print(intrinsic)
        #intrinsic = torch.tensor([intrinsic.intrinsic_matrix]).to(device)
        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj=camera
        #camera_traj=np.array(camera_traj).cpu()
        return camera_traj

class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        patch_size = [float('inf'), float('inf')]
        self.render = partial(render, pipe=pipe, bg_color=background, patch_size=patch_size)
        #self.pipe = pipe
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.viewpoint_stack = []

    
    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            
            #patch_size = [float('inf'), float('inf')]
            #background = torch.rand((3), dtype=torch.float32, device="cuda")
            #render_pkg = self.render(viewpoint_cam, self.gaussians, self.pipe, background, patch_size)
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            #alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['normal'], dim=0)
            depth = render_pkg['depth']
            #depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            #self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            #self.depth_normals.append(depth_normal.cpu())
        
        self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        self.depthmaps = torch.stack(self.depthmaps, dim=0)
        #self.alphamaps = torch.stack(self.alphamaps, dim=0)
        #self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        #print(len(mask_image))
        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            #print(cam_o3d)
            #print("mak_image[i]",mask_image[i])
            #print(image_path)
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            #print(depth.shape)
            #print(self.depthmaps)
            #mask_image=cv2.resize(mask_image,(1600,900))
           
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                #print(self.viewpoint_stack[i].gt_alpha_mask)
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 1)] = 0
            '''
            print(mask_image)
            
            for y in range(depth.shape[1]):
                for x in range(depth.shape[2]):
                    if mask_image[y, x] == 0:  # 如果掩模图片中该位置为黑色
                        depth[0,y, x] = 0
            '''        		
            print('',depth.shape)
            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )
            #print(depth.shape)
            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
      
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, normalmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sampled_normal = torch.nn.functional.grid_sample(normalmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, sampled_normal, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                samples = inv_contraction(samples)
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, normal, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    normalmap = self.depth_normals[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
