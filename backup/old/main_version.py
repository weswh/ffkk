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

import src.file_io as fio
import src.geometry3D as g3d
import src.geometry2D as g2d
import src.foot_params_verion2 as foot
from segments.predict import Predict

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(name)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger('MESSAGE')

class Feet(object):
    def __init__(self, uid: str, debug: str, measure: str) -> None:
        self.uid = uid or ''
        self.measure = measure or 'all'

        self.debug = debug or 'False'
        
        self.root_path = "./0827"
        self.data_path = os.path.join(self.root_path, self.uid)
        
        self.batch_sizes = 5
        self.yaml_path = "./segments/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml"
        self.feet_model_path = "./segments/model/model_final.pth"

        self.pose_path = os.path.join(self.data_path, "transforms.json")
        self.mesh_path = os.path.join(self.data_path, "mesh.ply")
        self.feet_path = os.path.join(self.data_path, "feet.obj")

        self.images_path = os.path.join(self.data_path, "images")
        self.u_images_path = os.path.join(self.data_path, "u_images")
        self.u_image_path = os.path.join(self.data_path, "u_image")
        self.u_masks_path = os.path.join(self.data_path, "u_masks")

        self.pressure_path = os.path.join(self.data_path, "pressure_plus.png")

    def run(self):
######### Step01 undistort images
        start_time = time.perf_counter()
        logger.info(f'Data path: {self.data_path}')
        logger.info('Start to undistort images...')

        g2d.undist_wide_cameras(self.images_path, self.pose_path)

        t2 = time.perf_counter()
        logger.info(f'Use time: {round(t2 - start_time, 2)}s')


######## Step02 segment feet
        logger.info('Start to segment feet...')

        images_name, images, _ = fio.read_images_from_folder(self.u_images_path)

        feet_pred = Predict(self.yaml_path, self.feet_model_path)
        feet_pred_masks = feet_pred(images, self.batch_sizes)
        feet_ids, feet_masks = g2d.select_feet_masks(feet_pred_masks)

        feet_masks_names = np.asarray([f"{n[:-4]}.png" for n in images_name])
        fio.save_images_to_dir(os.path.join(self.u_masks_path, "select"), 
                feet_masks_names[feet_ids], feet_masks[feet_ids])

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


# ######### Step03 mesh
#         logger.info('Start reconstruction...')
     
#         self.data_path=self.data_path+'/'
#         print(f"{self.data_path}")
#         subprocess.call(["bash", "/home/chuck/gaussian_surfels/test.sh", f"{self.data_path}"])
        
#         t4 = time.perf_counter()
#         logger.info(f'Use time: {round(t4 - t3, 2)}s')


# ######### Step04 measure
#         logger.info('Start to calculate feet size with mesh...')

#         mesh = o3d.io.read_triangle_mesh(self.mesh_path)
#         if self.debug == 'True':
#             g2d.reproject_mesh(mesh, self.u_images_path, self.pose_path)
#             g2d.reproject_mesh(mesh, os.path.join(self.data_path, "eval_mask"), self.pose_path, outputs_name="eval")

#         mesh_x, _ = g3d.rotate_mesh_with_axis(mesh, -90, "x")
#         mesh_xy, _ = g3d.rotate_mesh_with_axis(mesh_x, 180, "y")
#         mesh_xy.scale(100, np.zeros(3))  # 脚的单位改成毫米

#         mesh_t = g3d.crop_mesh(mesh, -20, 100)
#         mesh_sort = g3d.sort_coordz(mesh_t)
#         o3d.io.write_triangle_mesh(self.feet_path, mesh_sort)
#         # g3d.capture_mesh_view(mesh_sort, os.path.join(self.data_path, "feet_mod.jpg"))
#         g2d.draw_report_image(self.data_path)

#         # 正交投影测量脚长脚宽
#         image_path = os.path.join(self.data_path, "obj.png")
#         g3d.orthographic_projection(mesh_sort, image_path)
#         g2d.render_contour(image_path)
#         left_width, left_length, right_width, right_length, area1, area2 \
#             = g3d.left_right_feet(mesh_sort, image_path)

#         length1 = int(round(left_length, 0))
#         length2 = int(round(right_length, 0))
#         total = length1 + length2
#         diff1 = abs(length2 - length1) * length1 / total
#         diff2 = abs(length2 - length1) * length2 / total

#         if length1 > length2:
#             left = length1 - diff1
#             right = length2 + diff2
#         else:
#             left = length1 + diff1
#             right = length2 - diff2

#         left_size = foot.calculate_shoe_size(left)
#         right_size = foot.calculate_shoe_size(right)

#         self.pressure_path = os.path.join(self.data_path, "pressure_plus.png")
#         gravity = foot.calculate_gravity(self.pressure_path)

#         mesh_rot, _ = g3d.rotate_mesh_with_axis(mesh_sort, 180, "x")
#         left_type, right_type = foot.overlap_mesh_press(mesh_rot, self.pressure_path)

#         if self.measure == "all":
#             ret0 = {"left": {"length":int(round(left, 0)), 
#                              "width": int(round(left_width, 0)), 
#                              "size": left_size,
#                              "is_flatfooted": left_type},
#                     "right":{"length": int(round(right, 0)), 
#                              "width": int(round(right_width, 0)), 
#                              "size": right_size,
#                              "is_flatfooted": right_type},
#                     "gravity": gravity}    
#             fio.save_json(os.path.join(self.data_path, "data0.json"), ret0)

#             mesh_sort, _ = g3d.rotate_mesh_with_axis(mesh_sort, 180, "x")
#             left_pcd, right_pcd = g3d.left_right_mesh(mesh_sort)
#             # o3d.io.write_triangle_mesh(os.path.join(self.data_path, "debug.ply"), mesh_sort)
#             arch_limg, hallux_limg, heel_limg, lresult = \
#                 foot.detect_foot_params(left_pcd, direction="left", length=left)
#             arch_rimg, hallux_rimg, heel_rimg, rresult = \
#                 foot.detect_foot_params(right_pcd, direction="right", length=right)
            
#             if self.debug == "True":
#                 left_image = os.path.join(self.data_path, "left")
#                 os.makedirs(left_image, exist_ok=True)
#                 cv.imwrite(os.path.join(left_image, "arch.jpg"), arch_limg)
#                 cv.imwrite(os.path.join(left_image, "hallux.jpg"), hallux_limg)
#                 cv.imwrite(os.path.join(left_image, "heel.jpg"), heel_limg)

#                 right_image = os.path.join(self.data_path, "right")
#                 os.makedirs(right_image, exist_ok=True)
#                 cv.imwrite(os.path.join(right_image, "arch.jpg"), arch_rimg)
#                 cv.imwrite(os.path.join(right_image, "hallux.jpg"), hallux_rimg)
#                 cv.imwrite(os.path.join(right_image, "heel.jpg"), heel_rimg)

#             ret = { "left": {"length": int(round(left, 0)), 
#                              "width": int(round(left_width, 0)), 
#                              "size": left_size,
#                              "is_flatfooted": left_type,
#                              "params": lresult}, 
#                     "right":{"length": int(round(right, 0)), 
#                              "width": int(round(right_width, 0)), 
#                              "size": right_size,
#                              "is_flatfooted": right_type,
#                              "params": rresult}, 
#                     "gravity": gravity}
            
#             fio.save_json(os.path.join(self.data_path, "data.json"), ret)
#             logger.info(f"Results: {ret}")
        
#         t5 = time.perf_counter()
#         logger.info('Caculate finish!')
#         logger.info(f'Use time: {round(t5 - t4, 2)}s')

#         end_time = time.perf_counter()
#         logger.info(f"Finish! Use total time: {round(end_time - start_time, 2)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="足扫")
    parser.add_argument("--uid", type=str, default="20240613/gaussion_surfels/CY/01", required=True)
    parser.add_argument("--measure", type=str, default="all")
    parser.add_argument("--debug", type=str, default='False')

    args = parser.parse_args()

    feet3d = Feet(args.uid, args.debug, args.measure)
    feet3d.run()