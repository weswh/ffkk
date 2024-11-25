import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, poisson_mesh
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, match_depth, resample_points, mask_prune, grid_prune, depth2viewDir, img2video
from utils.graphics_utils import getProjectionMatrix
from utils.camera_utils import interpolate_camera
from argparse import ArgumentParser
from torchvision.utils import save_image
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from torch.utils.cpp_extension import load
import pymeshlab
import time
import open3d as o3d
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh


def render_set(image_path, model_path, use_mask, name, iteration, views, gaussians, pipeline, background, write_image, poisson_depth, depth_trunc, voxel_size, sdf_trunc):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    info_path = os.path.join(model_path, name, "ours_{}".format(iteration), "info")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(info_path, exist_ok=True)

    if name == 'train':
        bound = None
        occ_grid, grid_shift, grid_scale, grid_dim = gaussians.to_occ_grid(0.0, 512, bound)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        background = torch.zeros((3), dtype=torch.float32, device="cuda")
        render_pkg = render(view, gaussians, pipeline, background, [float('inf'), float('inf')])
        
        image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(image.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )
        view=to_cam_open3d(view)
        volume.integrate(rgbd, intrinsic=view.intrinsic, extrinsic=view.extrinsic)


    if name == 'train':
        mesh_path = f'{model_path}/poisson_mesh_{poisson_depth}.ply'  
        mesh = volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        view=views[0]
        mesh_post = post_process_mesh(mesh,image_path, cluster_to_keep=100)
        #mesh_smoothed = mesh_post.filter_smooth_taubin(number_of_iterations=100) 
        o3d.io.write_triangle_mesh(mesh_path.replace('.ply', '_post.ply'), mesh_post.filter_smooth_laplacian(1).filter_smooth_laplacian(1) )




def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, write_image: bool, poisson_depth: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)

        scales = [1]
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=scales)
        mesh_res = 1024
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussExtractor = GaussianExtractor(gaussians, render, pipeline, bg_color=bg_color)  
        gaussExtractor.reconstruction(scene.getTrainCameras())
        #设置基础，深度范围，提速大小，sdf阈值等信息
        # voxel_size=0.004 
        # sdf_trunc=0.06
        #depth_trunc=20
        #print('@@@@@@@',dataset.source_path)
        depth_trunc = (gaussExtractor.radius * 2.0) 
        voxel_size = (depth_trunc / mesh_res) 
        sdf_trunc = 5.0 * voxel_size
        if not skip_test:
             render_set(dataset.source_path,dataset.model_path, True, "test", scene.loaded_iter, scene.getTestCameras(scales[0]), gaussians, pipeline, background, write_image, poisson_depth, depth_trunc, voxel_size, sdf_trunc)

        if not skip_train:
             render_set(dataset.source_path,dataset.model_path, True, "train", scene.loaded_iter, scene.getTrainCameras(scales[0]), gaussians, pipeline, background, write_image, poisson_depth, depth_trunc, voxel_size, sdf_trunc)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--img", action="store_true")
    parser.add_argument("--depth", default=10, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.img, args.depth)