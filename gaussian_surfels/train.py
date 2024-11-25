#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import torch
from utils.graphics_utils import patch_offsets, patch_warp
from random import randint
from utils.loss_utils import l1_loss, ssim, cos_loss, bce_loss, knn_smooth_loss , lncc
from gaussian_renderer import render, network_gui
import numpy as np
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, match_depth, normal2curv, resize_image, cross_sample
from torchvision.utils import save_image
from argparse import ArgumentParser, Namespace
import time
import os
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
# import dlib
# from gaustudio.utils.graphics_utils import depth_to_normal
#from gaustudio.losses import l1_loss, ssim
from scene.app_model import AppModel
from PIL import Image


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, opt.camera_lr, shuffle=False, resolution_scales=[1, 2, 4])
    use_mask = dataset.use_mask
    gaussians.training_setup(opt)
    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif use_mask: # visual hull init
        gaussians.mask_prune(scene.getTrainCameras(), 4)
        None
    
    opt.densification_interval = max(opt.densification_interval, len(scene.getTrainCameras()))

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    pool = torch.nn.MaxPool2d(9, stride=1, padding=4)


    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    count = -1
    #print(opt.iterations)
    for iteration in range(first_iter, opt.iterations + 2):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if iteration - 1 == 0:
            scale = 4
        elif iteration - 1 == 3000:
            scale = 2
        elif iteration - 1 >= 5000:
            scale = 1
        # scale = 1
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale).copy()[:]
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        viewpoint_cam = viewpoint_stack.pop(0)
        # viewpoint_cam = scene.getTrainCameras(scale)[0]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        background = torch.rand((3), dtype=torch.float32, device="cuda") if dataset.random_background else background
        patch_size = [float('inf'), float('inf')]
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, patch_size)
        image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        mask_gt = viewpoint_cam.get_gtMask(use_mask)
        #print('mask:',mask_gt)
  
        gt_image = viewpoint_cam.get_gtImage(background, use_mask)
        # print(gt_image.shape)
        gt_image_np = gt_image.permute(1, 2, 0).cpu().numpy()

        # 将 NumPy 数组转换为 PIL 图像
        gt_image_pil = Image.fromarray((gt_image_np * 255).astype(np.uint8))

        # 将图像转换为灰度图像
        gt_image_gray = gt_image_pil.convert('L')
        # gt_image_gray = gt_image.cpu().convert('L')
        #print('gt_image',gt_image)
        #print(gt_image)
        mask_vis = (opac.detach() > 1e-5)
        #print(mask_vis)
        normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
        d2n = depth2normal(depth, mask_vis, viewpoint_cam)
        mono = viewpoint_cam.mono if dataset.mono_normal else None
        #print(mono)
        if mono is not None:
            mono *= mask_gt
            monoN = mono[:3]
            #monoD = mono[3:]
            #monoD_match, mask_match = match_depth(monoD, depth, mask_gt * mask_vis, 256, [viewpoint_cam.image_height, viewpoint_cam.image_width])
        # face_loss=0
        # for face in viewpoint_cam.faces:
        #     (x, y, w, h) = face.left(), face.top(), face.width(), face.height()
        #     face_loss =l1_loss(image[x:x+w,y:y+h],gt_image[x:x+w,y:y+h])
        #     if iteration %100 ==0:
        #         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #         cv2.imwrite('/home/veily/gaussian_surfels/data/zxj/old_child/scan2/'+'output_image_with_boxes'+str(iteration)+'.jpg', image)
        Ll1 = l1_loss(image, gt_image)
        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss_mask = (opac * (1 - pool(mask_gt))).mean()
        if mono is not None:
            loss_monoN = cos_loss(normal, monoN, weight=mask_gt)
            #loss_depth = l1_loss(depth * mask_match, monoD_match)
        loss_surface = cos_loss(resize_image(normal, 1), resize_image(d2n, 1), thrsh=np.pi*1/10000 , weight=1)
        opac_ = gaussians.get_opacity - 0.5
        opac_mask = torch.gt(opac_, 0.01) * torch.le(opac_, 0.99)
        loss_opac = torch.exp(-(opac_ * opac_) * 20)
        loss_opac = (loss_opac * opac_mask).mean()
        curv_n = normal2curv(normal, mask_vis)
        curv_d2n = normal2curv(d2n, mask_vis)
    #     depth_normal = depth_to_normal(viewpoint_cam, depth[None, ...])
    #    # print('depth_normal',depth_normal)
    #     depth_normal = depth_normal.permute(0, 3, 1, 2)
        render_normal = torch.nn.functional.normalize(normal, p=2, dim=0)
        c2w = (viewpoint_cam.world_view_transform.T).inverse()
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        #print('render_normal_world:',render_normal_world)
        normal_error = 1 - (render_normal_world * d2n).sum(dim=0)
        depth_normal_loss = normal_error.mean()
        # bg = torch.rand((3), device="cuda") if opt.random_background else background
        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, background, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                #将点转换到最近相机坐标系
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                #转换坐标后渲染到当前图像并计算误差
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                d_mask = d_mask & (pixel_noise < pixel_noise_th)
                #基于投影噪声计算权重，噪声大的权重小
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0
                #调试信息输出，渲染图像、法线、深度图及权重等的可视化
                if iteration % 200 == 0:
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((d2n +1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    row0 = np.concatenate([gt_img_show, img_show, normal_show], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = rendered_normal.reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(H,W)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

        loss_curv = l1_loss(curv_n * 1, 0) + 1 * l1_loss(curv_d2n, 0)
        
        loss = 1 * loss_rgb #+ face_loss
        loss += 1 * loss_mask
        loss += (0.01 + 0.1 * min(2 * iteration / opt.iterations, 1)) * loss_surface
        #loss += (0.00 + 0.1 * min(1 * iteration / opt.iterations, 1)) * loss_surface
        loss += (0.005 - ((iteration / opt.iterations)) * 0.0) * loss_curv 
        loss += loss_opac * 0.01
        loss +=(0.005 - ((iteration / opt.iterations)) * 0.0) * depth_normal_loss
        # lambda_depth_normal = opt.lambda_depth_normal if iteration >= opt.depth_normal_from_iter else 0.0
        # # mono = None
        # loss += lambda_depth_normal
        if mono is not None:
            loss += (0.04 - ((iteration / opt.iterations)) * 0.03) * loss_monoN * 5
            #loss += 0.01 * loss_depth 
        loss.backward()

        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_rgb.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, Pts={len(gaussians._xyz)}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            test_background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, pipe, test_background, use_mask)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            #Densification
            if iteration > opt.densify_from_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                min_opac = 0.1# if iteration <= opt.densify_from_iter else 0.1
                # min_opac = 0.05 if iteration <= opt.densify_from_iter else 0.005
                # if iteration % opt.pruning_interval == 0:
                
                if iteration % opt.densification_interval == 0:
                    gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                    gaussians.adaptive_densify(opt.densify_grad_threshold, scene.cameras_extent)
                
                if (iteration - 1) % opt.opacity_reset_interval == 0 and opt.opacity_lr > 0:
                    gaussians.reset_opacity(0.12, iteration)
            
            if (iteration - 1) % 1000 == 0:

                if mono is not None:
                    monoN_wrt = normal2rgb(monoN, mask_gt)
                    # monoD_wrt = depth2rgb(monoD_match, mask_match)
                normal_wrt = normal2rgb(normal, mask_vis)
                depth_wrt = depth2rgb(depth, mask_vis)
                img_wrt = torch.cat([gt_image, image, normal_wrt * opac, depth_wrt * opac], 2)
                save_image(img_wrt.cpu(), './gaussian_surfels/testrenderme/test'+str(iteration)+'.png')         
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad()
                # viewpoint_cam.optimizer.step()
                # viewpoint_cam.optimizer.zero_grad()
            if (iteration in checkpoint_iterations):
                # gaussians.adaptive_prune(min_opac, scene.cameras_extent)
                # scene.visualize_cameras()
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())

        args.model_path = os.path.join("./gaussian_surfels/outputs/")#_{unique_str[0:10]}")
        
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, pipe, bg, use_mask):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()[::8]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(render(viewpoint, scene.gaussians, pipe, bg, [float('inf'), float('inf')])["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.get_gtImage(bg, with_mask=use_mask), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 12_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000, 12_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
