import os
import argparse
import shutil
import warnings
import time
import open3d as o3d
import numpy as np
import cv2 as cv
import logging
import subprocess
import shutil
import sys

import src.file_io as fio
import src.geometry3D as g3d
import src.geometry2D as g2d
import src.foot_params_verion2 as foot
from segments.predict import Predict
import yaml
import src.Measure as msr
import src.pressure_output as pro


os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

logging.basicConfig(level=logging.INFO,stream=sys.stdout, format='[%(asctime)s %(name)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger('MESSAGE')

class Feet(object):
    def __init__(self, uid: str, debug: str, measure: str, config_path="./config/config.yaml") -> None:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        # 从配置文件中加载路径和参数
        paths = config['paths']
        params = config['parameters']
        # 初始化成员变量
        self.uid = uid or ''
        self.measure = measure or 'all'
        self.debug = debug or 'False'
        # 路径配置
        self.yaml_path = paths['yaml_path']
        self.feet_model_path = paths['feet_model_path']
        self.root_path = paths['root_path']
        self.data_path = os.path.join(self.root_path, self.uid)
        self.pose_path = os.path.join(self.data_path, "transforms.json")
        self.mesh_path = os.path.join(self.data_path, "mesh.ply")
        self.left_mesh_path = paths['left_mesh_path']
        self.right_mesh_path = paths['right_mesh_path']
        self.pred_left_mesh_path = paths['pred_left_mesh_path']
        self.pred_right_mesh_path = paths['pred_right_mesh_path']
        self.feet_path = os.path.join(self.data_path, "feet.obj")
        self.images_path = os.path.join(self.data_path, "images")
        self.u_images_path = os.path.join(self.data_path, "u_images")
        self.u_image_path = os.path.join(self.data_path, "u_image")
        self.u_masks_path = os.path.join(self.data_path, "u_masks")
        self.pressure_path = os.path.join(self.data_path, "pressure_plus.png")
        self.file_path = './config/outputs.xlsx'
        # 参数配置
        self.batch_sizes = params['batch_sizes']
        self.min_contour_area = params['min_contour_area']
        self.min_valid_masks = params['min_valid_masks']
    def run(self):
######## Step01 undistort images
        start_time = time.perf_counter()
        logger.info(f'Data path: {self.data_path}')
        logger.info('Start to undistort images...')
        # g2d.rename_images_in_folder(self.images_path)
        g2d.undist_wide_cameras(self.images_path, self.pose_path)
        t2 = time.perf_counter()
        logger.info(f'Use time: {round(t2 - start_time, 2)}s')

######## Step02 segment feet
        logger.info('Start to segment feet...')
        images_name, images, _ = fio.read_images_from_folder(self.u_images_path)
        #通过预训练好的模型分割图像
        feet_pred = Predict(self.yaml_path, self.feet_model_path)
        feet_pred_masks = feet_pred(images, self.batch_sizes)
        feet_ids, feet_masks = g2d.select_feet_masks(feet_pred_masks,self.min_contour_area,self.min_valid_masks)
        feet_masks_names = np.asarray([f"{n[:-4]}.png" for n in images_name])
        #掩模图像在u_masks/select文件夹里
        fio.save_images_to_dir(os.path.join(self.u_masks_path, "select"), 
                feet_masks_names[feet_ids], feet_masks[feet_ids])
        #debug模式，检测分割等是否有错误
        if self.debug == 'True':
            fio.save_images_to_dir(os.path.join(self.u_masks_path, "raw"), 
                            feet_masks_names, feet_pred_masks)
            fio.save_images_to_dir(os.path.join(self.u_masks_path, "filter"), 
                            feet_masks_names, feet_masks)
            g2d.eval_mask(os.path.join(self.u_masks_path, "select"), self.u_images_path)
        
        u_image_names = np.asarray([f"{n[:-4]}.png" for n in images_name])
        u_images = g2d.rgbs_masks_to_ngp(images[feet_ids], feet_masks[feet_ids])
        fio.save_images_to_dir(self.u_image_path, u_image_names[feet_ids], u_images)
        t3 = time.perf_counter()
        logger.info(f'Use time: {round(t3 - t2, 2)}s')

######### Step03 Train mesh
        logger.info('Start reconstruction...')
        self.data_path=self.data_path
        print(f"{self.data_path}")
        subprocess.call(["bash", "./gaussian_surfels/test.sh", f"{self.data_path}"])
        t4 = time.perf_counter()
        logger.info(f'Use time: {round(t4 - t3, 2)}s')
        end_time = time.perf_counter()
        logger.info(f"Finish! Use total time: {round(end_time - start_time, 2)}s")

######## Step04 Predict mesh
        logger.info('Start pred mesh...')
        fio.filter_and_save_left_model(self.mesh_path,self.left_mesh_path)
        fio.filter_and_save_right_model(self.mesh_path,self.right_mesh_path)
        subprocess.call(["bash", "./FIND/test.sh",f"{self.mesh_path}"])

######### Step05 measure
        logger.info('Start to calculate feet size with mesh...')
     
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        
        if self.debug == 'True':
            g2d.reproject_mesh(mesh, self.u_images_path, self.pose_path)
            g2d.reproject_mesh(mesh, os.path.join(self.data_path, "eval_mask"), self.pose_path, outputs_name="eval")

        mesh_x, _ = g3d.rotate_mesh_with_axis(mesh, -90, "x")
        mesh_xy, _ = g3d.rotate_mesh_with_axis(mesh_x, 180, "y")
        mesh_xy.scale(100, np.zeros(3))  # 脚的单位改成毫米

        mesh_t = g3d.crop_mesh(mesh, -20, 100)
        mesh_sort = g3d.sort_coordz(mesh_t)
        o3d.io.write_triangle_mesh(self.feet_path, mesh_sort)
        # g3d.capture_mesh_view(mesh_sort, os.path.join(self.data_path, "feet_mod.jpg"))
        g2d.draw_report_image(self.data_path)

        # 投影测量脚长脚宽
        image_path = os.path.join(self.data_path, "obj.png")
        # g3d.orthographic_projection(mesh_sort, image_path)
        g3d.draw_picture(mesh_sort, image_path)
        g2d.render_contour(image_path)
        # left_width, left_length, right_width, right_length, area1, area2 \
        #     = g3d.left_right_feet(mesh_sort, image_path)
        left_width, left_length, left_farthest, right_width, right_length ,right_farthest\
            = g3d.left_right_feet_trans(mesh_sort, image_path)
       
        left = int(round(left_length, 0))
        right = int(round(right_length, 0))
        left_size = foot.calculate_shoe_size(left)
        right_size = foot.calculate_shoe_size(right)

        # 根据模型输出：
        left_lengths,left_widths,left_heights = msr.Calculating_model_parameters(left_farthest, self.pred_left_mesh_path)
        right_lengths,right_widths,right_heights = msr.Calculating_model_parameters(right_farthest, self.pred_right_mesh_path)

        #根据压力图输出：
        self.pressure_path = os.path.join(self.data_path, "pressure.png")

        # left_type, right_type = foot.overlap_mesh_press(mesh_rot, self.pressure_path)
        left_type, right_type = g2d.compute_flatfoot(self.pressure_path)
        left_pcd, right_pcd = g3d.left_right_mesh(mesh_sort)
        # o3d.io.write_triangle_mesh(os.path.join(self.data_path, "debug.ply"), mesh_sort)
        #足弓高度等
        arch_limg, hallux_limg, lresult = \
            foot.detect_foot_params(left_pcd, direction="left", length=left)
        arch_rimg, hallux_rimg,rresult = \
            foot.detect_foot_params(right_pcd, direction="right", length=right)
        self.pro_path = os.path.join(self.data_path, "obj2.png")
        left_angle, right_angle, left_type_hullux, right_type_hullux = foot.compute_hallux_valgus(self.pro_path)
        pro.compute_pressure_param(self.pressure_path,self.data_path)

        left_angle, right_angle = g2d.compute_Hallux_valgus(os.path.join(self.data_path, "obj2.png"))

        if self.debug == "True":
            left_image = os.path.join(self.data_path, "left")
            os.makedirs(left_image, exist_ok=True)
            cv.imwrite(os.path.join(left_image, "arch.jpg"), arch_limg)
            cv.imwrite(os.path.join(left_image, "hallux.jpg"), hallux_limg)
            # cv.imwrite(os.path.join(left_image, "heel.jpg"), heel_limg)

            right_image = os.path.join(self.data_path, "right")
            os.makedirs(right_image, exist_ok=True)
            cv.imwrite(os.path.join(right_image, "arch.jpg"), arch_rimg)
            cv.imwrite(os.path.join(right_image, "hallux.jpg"), hallux_rimg)
            # cv.imwrite(os.path.join(right_image, "heel.jpg"), heel_rimg)

        ret = { "left": {"length":int(round(left_farthest,0)), 
                            "width": int(round(left_width, 0)), 
                            "size": left_size,
                            "lengths": left_lengths,
                            "widths": left_widths,
                            "heights": left_heights,
                            "is_flatfooted": left_type,
                            "params": lresult,
                            "hallux_angle":left_angle*360/6.28,
                            "is_hallux_valgused":left_type_hullux
                        }, 
                "right":{ "length":  int(round(right_farthest,0)), 
                            "width": int(round(right_width, 0)), 
                            "size": right_size,
                            "lengths": right_lengths,
                            "widths": right_widths,
                            "heights": right_heights,
                            "is_flatfooted": right_type,
                            "params": rresult,
                            "hallux_angle":right_angle*360/6.28,
                            "is_hallux_valgused":right_type_hullux
                        }}
        
        fio.save_json(os.path.join(self.data_path, "data.json"), ret)

        
        t5 = time.perf_counter()
        logger.info('Caculate finish!')
        # logger.info(f'Use time: {round(t5 - t4, 2)}s')

    
        # end_time = time.perf_counter()
        # logger.info(f"Finish! Use total time: {round(end_time - start_time, 2)}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="足扫")
    parser.add_argument("--uid", type=str, default="20240613/gaussion_surfels/CY/01", required=True)
    parser.add_argument("--measure", type=str, default="all")
    parser.add_argument("--debug", type=str, default='False')

    args = parser.parse_args()

    feet3d = Feet(args.uid, args.debug, args.measure)
    feet3d.run()
