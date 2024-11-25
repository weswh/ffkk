from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from PIL import Image
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="足扫")
parser.add_argument('--root_path',
                default='/media/veily/CE72-49BD/shuju/shuju/',
                                        type=str, help='Project dir.')
args = parser.parse_args()
# model_path = os.path.join(args.root_path,'InstantSplat/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
model_path = "./dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
device = 'cuda'
batch_size = 4
schedule = 'linear'
lr = 0.01
niter = 100
C2W = []
K = []


def storePly(path, xyz, rgb, normal=None):
    # Define the dtype for the structured array  zh
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz) if normal is None else normal

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


#  自定义函数，用于读取自己标定的文件
def read_R_t(path, json_files_path,image_files):

    with open(json_files_path, 'r') as f:
        data = json.load(f)
    frames = data["frames"]
    # total_frames = len(frames)
    # step = total_frames // 5
    # selected_frames = frames[::step]
    # indices = [0, 2, 3, 5, 14, 21]
    #ep32
    indices = [0,4,6,9,13,16,18]#22张
    # indices = [0,3,5,8,9,11,13]#17张
    selected_frames = [frames[i] for i in indices]
    for i, frame in enumerate(selected_frames):
        # 1920*1080 -> 512 288
        #注意需要根据不同的图像尺寸进行修改
        # scale_x = round(288 / 1080, 2)
        # scale_y = round(512 / 1920, 2)
        #ep32
        scale_x = round(384 / 1536, 2)
        scale_y = round(512 / 2048, 2)
        frame["fl_x"] *= scale_x
        frame["fl_y"] *= scale_y
        frame["cx"] *= scale_x
        frame["cy"] *= scale_y
        K.append(np.array([[frame["fl_x"], 0, frame["cx"]],
                           [0, frame["fl_y"], frame["cy"]],
                           [0, 0, 1]]))
        # -----------根据儿童扫描设备针对性修改---------------
        # 图像缩放后，对内参进行缩放
        c2w = np.array(frame['transform_matrix'])
        # c2w = np.linalg.inv(c2w)
        # c2w = frame['transform_matrix']
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])  #坐标系互相转换 opencv->opengl或者opengl->opencv
        c2w = np.matmul(c2w, flip_mat)
        C2W.append(c2w)
        cam_name = os.path.join(path, frame["file_path"])
        image_path = f'{path}/u_image/'+frame["file_path"].split('/')[-1]#+extension
        # image_path = image_path.replace('png','jpg')
        # print(image_path)
        image_files.append(os.path.join(image_path))
        # image_name = Path(cam_name).stem
        # image = Image.open(image_path)
    return C2W, K, image_files


# ------------------注意事项--------------
# 功能：利用自己标定的内外参进行点云初始化
# 需要修改 root_path,以及列表image_number，需要用多少张图像进行训练，就初始化多少个True(建议20张以内)

if __name__ == "__main__":

    # root_path = "/mnt/newdisk/data/手机拍/视频素材（含数据文件）/谢良勇/苹果拍摄/455035f224d588fab1d52b4afec2515a_raw"  # 改一下路径
    root_path = args.root_path
    json_path = os.path.join(root_path, "transforms.json")
    # image_number = [True, True, True, True, True,True]
    #ep32
    image_number = [True, True, True, True, True, True, True]
    image_dir = os.path.join(root_path, "u_image")  #  图像路径
    save_dir = os.path.join(root_path, "outputs")# 保存路径
    #print('##########',save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # ！！！！！！！！！读取内外参，注意修改缩放系数！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！1
    image_files = []
    c2w, K, image_files = read_R_t(root_path,json_path,image_files)
    # Use os.listdir to get all files in the directory
    # for file in os.listdir(image_dir):
    #     image_files.append(os.path.join(image_dir, file))
    #print(image_files)
    image_files.sort(key=str.lower)
    model = load_model(model_path, device)
    # load_images can take a list of images or a directorypoints3D2
    images = load_images(image_files, size=512)
    # images = load_images(image_files, size=1920)#原来
    # images = load_images(image_files, size=960)

    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    print(output.keys())

    # print(*(output['view1']['img'])[0].shape[1:])
    # scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    # loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # 指定内外参
    # # here data in list of (intrinsics, pose)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, fx_and_fy=True)
    scene.preset_pose(c2w, image_number)
    scene.preset_intrinsics(K, image_number)
    # scene.preset_focal(focal,[True, True, True, True, True, True, True, True, True])

    loss = scene.compute_global_alignment(init="known_poses", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    K = scene.get_intrinsics()
    depths = scene.get_depthmaps()
    conf_vals = scene.get_conf()

    import torch

    torch.save(imgs, f'{save_dir}/imgs.pt')
    torch.save(focals, f'{save_dir}/focals.pt')
    torch.save(poses, f'{save_dir}/poses.pt')
    torch.save(pts3d, f'{save_dir}/pts3d.pt')
    torch.save(confidence_masks, f'{save_dir}/confidence_masks.pt')
    torch.save(K, f'{save_dir}/intrinsics.pt')
    torch.save(depths, f'{save_dir}/depths.pt')
    torch.save(conf_vals, f'{save_dir}/conf_vals.pt')

    # 直接保存点云
    ply_path = os.path.join(save_dir, 'points3D.ply')
    pt3d_path = os.path.join(save_dir, 'pts3d.pt')
    masks_path = os.path.join(save_dir, 'confidence_masks.pt')
    imgs_path = os.path.join(save_dir, 'imgs.pt')
    pointmaps = torch.load(pt3d_path)
    masks = torch.load(masks_path)
    imgs = torch.load(imgs_path)
    pointmaps = [p.detach().cpu().numpy() for p in pointmaps]
    masks = [m.detach().cpu().numpy() for m in masks]
    xyz = np.concatenate([p[m] for p, m in zip(pointmaps, masks)])
    rgb = np.concatenate([p[m] for p, m in zip(imgs, masks)])
    storePly(ply_path, xyz, rgb)