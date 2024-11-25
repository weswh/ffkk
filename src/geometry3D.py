import copy
import numpy as np
import cv2 as cv
import src.geometry2D as g2d
import src.file_io as fio
# import src.visualization as vis
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
# from alpha_shapes.alpha_shapes import Alpha_Shaper
# import shapely
import json
from pathlib import Path
import os
import cv2
import trimesh
import pyrender
from sklearn.cluster import KMeans
os.environ[ "PYOPENGL_PLATFORM" ] = "osmesa"

# 裁剪mesh
def crop_mesh(mesh, low, top):
    aabb = mesh.get_axis_aligned_bounding_box()
    max_bnd = aabb.get_max_bound()
    min_bnd = aabb.get_min_bound()
    max_bnd[2] = top
    min_bnd[2] = low
    aabb.max_bound = max_bnd
    aabb.min_bound = min_bnd
    mesh_ = copy.deepcopy(mesh).crop(aabb)
    return mesh_

def crop_mesh_axis(mesh, low, top, axis):
    aabb = mesh.get_axis_aligned_bounding_box()
    max_bnd = aabb.get_max_bound()
    min_bnd = aabb.get_min_bound()
    if axis == "x":
        max_bnd[0] = top
        min_bnd[0] = low
    elif axis == "y":
        max_bnd[1] = top
        min_bnd[1] = low
    elif axis == "z":
        max_bnd[2] = top
        min_bnd[3] = low
    aabb.max_bound = max_bnd
    aabb.min_bound = min_bnd
    mesh_ = copy.deepcopy(mesh).crop(aabb)
    return mesh_

def get_dst_pts(center_x,rect,width,height):
    if(center_x<250):
        dst_pts = np.array([[rect[0][0]-width/2, rect[0][1]-height/2],
                    [rect[0][0]+width/2, rect[0][1]-height/2],
                    [rect[0][0]+width/2, rect[0][1]+height/2],
                    [rect[0][0]-width/2, rect[0][1]+height/2]], dtype="float32")
    else:
        dst_pts = np.array([[rect[0][0]-width/2, rect[0][1]+height/2],
                    [rect[0][0]-width/2, rect[0][1]-height/2],
                    [rect[0][0]+width/2, rect[0][1]-height/2],
                    [rect[0][0]+width/2, rect[0][1]+height/2]], dtype="float32")
    return dst_pts

def get_farthest_distance(warped,box,save_path):
    white_points = np.argwhere(warped > 60)
    # print(white_points[-1])
    y_min = white_points[white_points[:, 0] == int(box[1][1])+1]  # 找到 y 最低点
    y_max = white_points[white_points[:, 0] == int(box[3][1])-1]  # 找到 y 最高点
    if len(y_min) > 0:
        avg_x1 = np.mean(y_min[:, 1])  # group1 的 x 坐标平均值
    else:
        print("点的数量不足以进行计算") # 如果没有找到符合条件的点，可以设置为 None 或其他默认值

    if len(y_max) > 0:
        avg_x2 = np.mean(y_max[:, 1])  # group2 的 x 坐标平均值
    else:
        print("点的数量不足以进行计算")
    point1 = (avg_x1, box[1][1])
    point2 = (avg_x2, box[3][1]) 
    print(point1,point2)    
    distance = np.sqrt((point1[1] - point2[1]) ** 2 + (point1[0] - point2[0]) ** 2)
    point1 = (int(avg_x1), int(box[1][1]))
    point2 = (int(avg_x2), int(box[3][1]))
    warped_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    cv2.circle(warped_color, point1, radius=3, color=(0, 0, 255), thickness=-1)  # 实心红点
    cv2.circle(warped_color, point2, radius=3, color=(0, 0, 255), thickness=-1)  # 实心红点

    if(box[0][0]<250):
        type = 'left'
    else:
        type = 'right'

    # 在图像上标注两点之间的距离
    cv2.putText(warped_color, f"Distance: {distance:.2f}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 可视化轮廓
    cv2.drawContours(warped_color, [box.astype(int)], -1, (0, 255, 0), 2)

    # 保存带标注的图像
    path = save_path.replace('obj.png','warped_trans_colored'+type+'.png')
    cv2.imwrite(path, warped_color)
    return distance

def count_length_width(images,save_path):
    contours, _ = cv2.findContours(images, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = contours[0]
    area= cv2.contourArea(contours[0])
    # cv2.fillPoly(images, [contour], (0, 0, 0))
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    #判断长宽
    if(rect[1][1]<rect[1][0]):
        width = int(rect[1][1])
        height = int(rect[1][0])
    else:
        width = int(rect[1][0])
        height = int(rect[1][1])
    src_pts = box.astype("float32")
    #获取转正后的方框坐标
    dst_pts = get_dst_pts(box[0][1],rect,width,height)
    # print(dst_pts)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(images, M, (images.shape[0], images.shape[1]))
    #在转正后的图像上画框，计算长宽
    # contours, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # contour = contours[0]
    # area= cv2.contourArea(contours[0])
    # # cv2.fillPoly(images, [contour], (0, 0, 0))
    # rect = cv2.minAreaRect(contour)
    # box = cv2.boxPoints(rect)

    farthest_distance = get_farthest_distance(warped,dst_pts,save_path)
    size = rect[1]
    if size[0]<size[1]:
        length = size[1]
        width = size[0]
    else:
        length = size[0]
        width = size[1]
    return length,width,farthest_distance


def left_right_feet_trans(mesh, save_path):
    verts = np.asarray(mesh.vertices)

    image = cv2.imread(save_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 200, 583, cv2.THRESH_BINARY_INV)
    # 将图像分为左半部分和右半部分,分别计算长宽
    image_left =  np.zeros_like(binary_image) 
    image_left [:,:250] =  binary_image[:,:250]
    length_left,width_left,farthest_left = count_length_width(image_left,save_path)

    image_right =  np.zeros_like(binary_image) 
    image_right[:, 250:] = binary_image[:,250:]
    length_right,width_right, farthest_right = count_length_width(image_right,save_path)

    return width_left,length_left, farthest_left, width_right, length_right,farthest_right
   

# 正射投影的计算方式
def left_right_feet(mesh, save_path):
    verts = np.asarray(mesh.vertices)

    image = cv2.imread(save_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 200, 583, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = contours[0], contours[1]
    area1= cv2.contourArea(contours[0])
    area2= cv2.contourArea(contours[1])

    size = []
    point = []

    for c in contour:
        cv2.fillPoly(image, [c], (0, 0, 0))
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        #保存长宽和中心点
        size.append(rect[1])
        # point.append(box)
        point.append(rect[0])
        cv2.drawContours(image, [box.astype(int)], -1, (0, 255, 0), 2)
        cv2.imwrite(save_path, image)
    
    # 对正交投影的图像做判断，x大的判定为右脚
    if point[0][0] > point[1][0]:

        right = size[0]
        left = size[1]
        right_area = area1
        left_area = area2
    else:
        right = size[1]
        left = size[0]
        right_area = area2
        left_area = area1

    # 分辨长短边，判断长度和宽度
    if left[0] > left[1]:
        left_length = left[0]
        left_width = left[1]
    else:
        left_width = left[0]
        left_length = left[1]

    if right[0] > right[1]:
        right_length = right[0]
        right_width = right[1]
    else:
        right_width = right[0]
        right_length = right[1]

    return left_width, left_length, right_width, right_length, left_area, right_area

#模型投影图
def draw_picture(mesh, save_path): 
    import copy
    mesh_0 = copy.deepcopy(mesh)
    verts = np.asarray(mesh_0.vertices)
    tris = np.asarray(mesh_0.triangles)

    # 图像尺寸
    image_width, image_height = 500, 500
    
    verts[:, 2] = 0
    x_coords = verts[:, 0]
    y_coords = verts[:, 1]
    # 找到 x 和 y 的范围
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # 计算模型的中心
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    # 计算图像中心
    image_x_center = image_width / 2
    image_y_center = image_height / 2

    x_offset = image_x_center - x_center
    y_offset = image_y_center - y_center

    # 平移顶点
    verts[:, 0] += x_offset
    verts[:, 1] += y_offset

    verts = np.rint(verts).astype(int)
    img = np.ones((image_width, image_height, 3), dtype=np.uint8) * 255
    img[verts[:,1],verts[:,0]] = [0,0,0]
    rotated_image = cv2.rotate(img, cv2.ROTATE_180)
    flipped_image = cv2.flip(rotated_image, 1)
    cv2.imwrite(save_path, flipped_image)



# 输出正交投影图
def orthographic_projection(mesh, save_path):
    mest_t = crop_mesh(mesh, -1, 60)
    verts = np.asarray(mesh.vertices)

    tris = np.asarray(mesh.triangles)
    mesh_tri = trimesh.Trimesh(vertices=verts, faces=tris)

    image_width, image_height = 500, 500

    z = mesh_tri.vertices[:,2]
    z_min, z_max = np.min(z), np.max(z)
    z = (z-z_min)/(z_max-z_min)
    mesh_tri.visual.vertex_colors = np.tile(z[:,np.newaxis], (1,3))

    x_mag, y_mag, z_mag = mesh_tri.bounding_box.primitive.extents
    x_off, y_off, z_off = mesh_tri.bounding_box.centroid

    mesh_pr = pyrender.Mesh.from_trimesh(mesh_tri)
    scene = pyrender.Scene()
    scene.add(mesh_pr)

    camera = pyrender.OrthographicCamera(xmag=image_width/2, ymag=image_height/2, zfar=z_mag+1, znear=0.01)
    camera_pose = np.array([[-1.0, 0.0, 0.0, x_off],
                            [0.0, -1.0, 0.0, y_off],
                            [0.0, 0.0, 1.0, 60.0],  # 100
                            [0.0, 0.0, 0.0, 1.0]])
    scene.add(camera, pose=camera_pose)

    r = pyrender.OffscreenRenderer(image_width, image_height, point_size=1.)  # 图像长宽要设置成一样，这样得到的xy的缩放比例也一样
    image, depth = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
    # cv2.imwrite(save_path, image)

    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite(save_path, rotated_image)
# 整理z<0的点，往上抬
def sort_coordz(mesh):
    for i in range(len(mesh.vertices)):
        vertex = mesh.vertices[i]
        if vertex[2] < -1:
            vertex[2] = 0
    return mesh

# mesh分左右脚，对于已经标定的设备
def left_right_mesh(mesh):
    verts = np.asarray(mesh.vertices)
    left = verts[verts[:, 0] > 0]  # x > 0 归为左脚
    right = verts[verts[:, 0] <= 0]  # x < 0 归为右脚

    left_pcd = o3d.geometry.PointCloud()
    left_pcd.points = o3d.utility.Vector3dVector(left)

    right_pcd = o3d.geometry.PointCloud()
    right_pcd.points = o3d.utility.Vector3dVector(right)

    return left_pcd, right_pcd

# 点云长边平行与y轴
def forward_pcd(pcd):
    obb = pcd.get_oriented_bounding_box()
    obb_rotation_matrix = obb.R  # 提取 OBB 的旋转矩阵
    obb_center = obb.center

    extents = obb.extent
    max_extent_idx = np.argmax(extents)  # 计算最长边的方向
    long_side_direction = obb_rotation_matrix[:, max_extent_idx]  # 获取obb最长边的方向向量

    y_axis = np.array([0, 1, 0])
    dot_product = np.dot(long_side_direction, y_axis)
    angle = np.arccos(dot_product / np.linalg.norm(long_side_direction))  # 计算长边与y轴的夹角

    if angle < (np.pi/2):
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle])
    elif angle > (np.pi/2):
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle-np.pi])
    pcd.rotate(R, center=obb_center)  

    return pcd

# mesh按照x、y、z轴旋转
def rotate_mesh_with_axis(mesh, angle, axis):
    theta = np.radians(angle)
    if axis == "x":
        rotation_matrix = np.array([[1, 0, 0, 0],
                                    [0, np.cos(theta), -np.sin(theta), 0],
                                    [0, np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 0, 1]])
    elif axis == "y":
        rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta), 0], 
                                    [0, 1, 0, 0], 
                                    [-np.sin(theta), 0, np.cos(theta), 0], 
                                    [0, 0, 0, 1]])
    elif axis == "z":
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                                    [np.sin(theta), np.cos(theta), 0, 0],
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, 1]])
    mesh.transform(rotation_matrix)
    
    return mesh, rotation_matrix

# 找二维点对应的三维点
def get_point3d(pcd, point2d, part):
    if part == "closest":  # 计算足弓的高度
        pcd_t = crop_mesh(pcd, 0, 20)
        pts = np.asarray(pcd_t.points)
        distances = np.sqrt((pts[:, 0] - (point2d[0]-250))**2 + (pts[:, 1] - (point2d[1]-250))**2)

        indices = np.argmin(distances)
        closest_points = pts[indices]  # 获取这些点的坐标
        point3d = closest_points
        # print(f"最接近点是: {point3d}")  # 最接近点  

    elif part == "y_zmax":  # 计算脚背的高度
        pts = np.asarray(pcd.points)
        distances = np.sqrt((pts[:, 1] - (point2d[1]-250))**2)

        indices = np.where(distances < 3)[0]  # 找到距离小于 3 的点的索引
        closest_points = pts[indices]
        point3d = closest_points[np.argmax(closest_points[:, 2])]
        # print(f"y值相同的最高点是: {point3d}")  # y值相同的z值最高点  

    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector([point3d])  # 单个点的坐标
    # point.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([pcd, point])

    return point3d

# pcd按照给定平面裁剪
def crop_pcd_with_plane(pcd, plane_origin, plane_normal, distance_threshold=0.01):
    points = np.asarray(pcd.points)
    distances = np.dot(plane_origin - points, plane_normal)  # 使用给定的平面定义一个点与平面的距离

    mask = distances > distance_threshold
    cropped_pcd = pcd.select_by_index(np.where(mask)[0])

    return cropped_pcd

# 输出模型的渲染图
def capture_mesh_view(mesh, output_path, zoom_factor=1.2):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    mesh_tri = trimesh.Trimesh(vertices=verts, faces=tris, vertex_colors=colors)

    bouding_box = mesh_tri.bounding_box_oriented
    center = bouding_box.centroid

    scene = pyrender.Scene()
    mesh_pr = pyrender.Mesh.from_trimesh(mesh_tri)
    scene.add(mesh_pr)

    view_params = {
    "eye": center + np.array([0.0, 300.0, 200.0]),
    "center": center,
    "up": np.array([0.0, -1.0, 1.0])
    }

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0 / zoom_factor)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = view_params['eye']
    camera_look_at = view_params['center'] - view_params['eye']
    camera_pose[:3, 2] = -camera_look_at / np.linalg.norm(camera_look_at)
    camera_pose[:3, 1] = view_params['up'] / np.linalg.norm(view_params['up'])
    camera_pose[:3, 0] = np.cross(camera_pose[:3, 1], camera_pose[:3, 2])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5) # 将光源的位置设置为与相机位置相同，以确保照亮模型
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080, point_size=1.0)
    color, depth = r.render(scene)
    cv2.imwrite(output_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))