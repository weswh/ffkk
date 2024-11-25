import pyvista as pv
import numpy as np
import yaml
with open('/media/liuyalan/Projects/足扫项目/FOOT/config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
coordinates = config['coordinates']
# def calculate_intersection_length(mesh, plane_coeffs):
#     # 使用PyVista的裁剪函数来获取交线
#     intersection = mesh.clip_plane(normal=plane_coeffs[:3], origin=plane_coeffs[3:])
    
#     # 获取交点的所有顶点
#     intersection_points = intersection.points

#     # 计算围长
#     perimeter_length = 0.0
#     for i in range(len(intersection_points)):
#         point1 = intersection_points[i]
#         point2 = intersection_points[(i + 1) % len(intersection_points)]  # 闭合环
#         perimeter_length += np.linalg.norm(point1 - point2)  # 计算相邻两点之间的距离

#     return perimeter_length


def calculate_plane_coeffs(point1, point2, point3):
    # 计算两个边
    vector1 = point2 - point1  # 从point1到point2的向量
    vector2 = point3 - point1  # 从point1到point3的向量
    
    # 计算法向量（向量1与向量2的叉乘）
    normal = np.cross(vector1, vector2)
    
    # 确保法向量是单位向量
    normal = normal / np.linalg.norm(normal)

    # 选择第一个点作为平面上的点
    point_on_plane = point1

    # plane_coeffs = [normal_x, normal_y, normal_z, point_x, point_y, point_z]
    plane_coeffs = np.concatenate((normal, point_on_plane))
    
    return plane_coeffs

def calculate_intersection_length(mesh, plane_coeffs):
    # 使用PyVista的裁剪函数来获取交线
    intersection = mesh.clip(normal=plane_coeffs[:3], origin=plane_coeffs[3:6],invert=False)
    
    # 获取交点的所有顶点
    intersection_points = np.array(intersection.points)

    # 计算相邻点之间的距离
    diff = np.diff(intersection_points, axis=0)
    distances = np.linalg.norm(diff, axis=1)

    # 计算围长（周长）
    perimeter_length = np.sum(distances)

    return perimeter_length

def point_to_plane_distance(point, plane_coeffs):
    # 提取平面方程的系数 a, b, c, d
    a, b, c, d = plane_coeffs
    point = np.array(point)
    
    # 计算点到平面的距离
    numerator = np.abs(a * point[0] + b * point[1] + c * point[2] + d)  # 计算分子
    denominator = np.sqrt(a**2 + b**2 + c**2)  # 计算分母
    
    distance = numerator / denominator
    return distance

# 计算点到直线的垂直投影
def project_point_onto_line(point, line_start, line_end):
    line_vector = line_end - line_start  # 计算直线向量
    line_vector_normalized = line_vector / np.linalg.norm(line_vector)  # 归一化直线向量
    
    # 计算点到直线起点的向量
    point_vector = point - line_start
    
    # 计算点在直线上投影的长度（垂直投影）
    projection_length = np.dot(point_vector, line_vector_normalized)  # 点到直线的投影长度
    
    # 计算投影点坐标
    projection_point = line_start + projection_length * line_vector_normalized  # 投影点
    
    return projection_point

def project_points_onto_line(points, line):
    # 遍历每个点，并调用 project_point_onto_plane 进行投影
    return np.array([project_point_onto_line(point, line[0],line[1]) for point in points])

# 计算两个点之间的欧几里得距离
def calculate_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # 计算两个点之间的欧几里得距离
    distance = np.linalg.norm(point1 - point2)
    
    return distance

#构建脚底
def create_plane(p1, p2, p3):
    # 将点表示为NumPy数组
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # 计算两个向量
    v1 = p2 - p1
    v2 = p3 - p1
    
    # 计算法向量，通过向量叉积
    normal_vector = np.cross(v1, v2)
    
    # 平面方程中的 (a, b, c)
    a, b, c = normal_vector
    
    # 计算平面方程中的 d
    d = -np.dot(normal_vector, p1)
    
    # 返回平面方程的系数 a, b, c, d
    return a, b, c, d

#投影点 
def project_point_onto_plane(point, plane_coeffs):
    # 提取平面方程的系数 a, b, c, d
    a, b, c, d = plane_coeffs
    point = np.array(point)
    
    # 平面法向量
    normal_vector = np.array([a, b, c])
    
    # 计算点到平面的距离
    distance = (np.dot(normal_vector, point) + d) / np.linalg.norm(normal_vector)
    
    # 计算投影点，沿着法向量的方向移动
    projection = point - distance * (normal_vector / np.linalg.norm(normal_vector))
    
    return projection

# 投影多个点到平面上
def project_points_onto_plane(points, plane_coeffs):
    # 遍历每个点，并调用 project_point_onto_plane 进行投影
    return np.array([project_point_onto_plane(point, plane_coeffs) for point in points])

def calculate_length(mesh,foot_plane,Midline_coords):
    length = {}
    #选取脚后跟点，大拇指及二拇值顶端，计算脚底长
    foot_bottom_length_coords = mesh.points[coordinates['foot_bottom_length_coords']]#[2504,42683,8]]
    foot_bottom_length_coords = project_points_onto_plane(foot_bottom_length_coords,foot_plane)
    foot_bottom_length_coords = project_points_onto_line(foot_bottom_length_coords,Midline_coords)
    foot_bottom_length1 = calculate_distance(foot_bottom_length_coords[0],foot_bottom_length_coords[2])
    foot_bottom_length2 = calculate_distance(foot_bottom_length_coords[1],foot_bottom_length_coords[2])
    length['foot_bottom_length'] = max(foot_bottom_length1,foot_bottom_length2)

    #足弓长
    Arch_coords = mesh.points[coordinates['Arch_coords']]#[43,12309]]
    Arch_coords = project_points_onto_plane(Arch_coords,foot_plane)
    Arch_coords = project_points_onto_line(Arch_coords,Midline_coords)
    length['Arch_length'] = calculate_distance(Arch_coords[0],Arch_coords[1])
    # print(Arch_length)
    #第一跖趾部位长度
    first_metatarsal_toe = mesh.points[coordinates['first_metatarsal_toe']]#[1776,12309]]
    first_metatarsal_coords = project_points_onto_plane(first_metatarsal_toe,foot_plane)
    first_metatarsal_coords = project_points_onto_line(first_metatarsal_toe,Midline_coords)
    length['first_metatarsal_length'] = calculate_distance(first_metatarsal_coords[0],first_metatarsal_coords[1])
    # print(first_metatarsal_length)

    #第五跖趾部位长度
    five_metatarsal_toe = mesh.points[coordinates['five_metatarsal_toe']]#[1717,12309]]
    five_metatarsal_coords = project_points_onto_plane(five_metatarsal_toe,foot_plane)
    five_metatarsal_coords = project_points_onto_line(five_metatarsal_toe,Midline_coords)
    length['five_metatarsal_length'] = calculate_distance(five_metatarsal_coords[0],five_metatarsal_coords[1])
    # print(five_metatarsal_length)

    #拇指外凸点长度
    # External_thumb_coords = mesh.points[[451,12309]]
    External_thumb_coords = mesh.points[coordinates['External_thumb_coords']]
    External_thumb_coords = project_points_onto_plane(External_thumb_coords,foot_plane)
    External_thumb_coords = project_points_onto_line(External_thumb_coords,Midline_coords)
    length['External_thumb_length'] = calculate_distance(External_thumb_coords[0],External_thumb_coords[1])
    # print(External_thumb_length)

    #舟上弯点长度
    Boat_bend_coords = mesh.points[coordinates['Boat_bend_coords']]#[671,12309]]
    Boat_bend_coords = project_points_onto_plane(Boat_bend_coords,foot_plane)
    Boat_bend_coords = project_points_onto_line(Boat_bend_coords,Midline_coords)
    length['Boat_bend_length'] = calculate_distance(Boat_bend_coords[0],Boat_bend_coords[1])
    # print(Boat_bend_length)

    #后脚根长度
    Hell_length_coords = mesh.points[coordinates['Hell_length_coords']]#[8,12309]]
    Hell_length_coords = project_points_onto_plane(Hell_length_coords ,foot_plane)
    Hell_length_coords = project_points_onto_line(Hell_length_coords ,Midline_coords)
    length['Hell_length'] = calculate_distance(Hell_length_coords[0],Hell_length_coords[1])
    # print(Hell_length)

    #跗骨突点长度
    Tarsal_coords = mesh.points[coordinates['Tarsal_coords']]#[4993,12309]]
    Tarsal_coords = project_points_onto_plane(Tarsal_coords,foot_plane)
    Tarsal_coords = project_points_onto_line(Tarsal_coords,Midline_coords)
    length['Tarsal_length'] = calculate_distance(Tarsal_coords[0],Tarsal_coords[1])
    # print(Tarsal_length)

    #外踝骨长度
    Outer_ankle_coords = mesh.points[coordinates['Outer_ankle_coords']]#[3100,12309]]
    Outer_ankle_coords = project_points_onto_plane(Outer_ankle_coords,foot_plane)
    Outer_ankle_coords = project_points_onto_line(Outer_ankle_coords,Midline_coords)
    length['Outer_ankle_length'] = calculate_distance(Outer_ankle_coords[0],Outer_ankle_coords[1])
    # print(Outer_ankle_length)

    #内踝骨长度
    Medial_ankle_coords = mesh.points[coordinates['Medial_ankle_coords']]#[496,12309]]
    Medial_ankle_coords = project_points_onto_plane(Medial_ankle_coords,foot_plane)
    Medial_ankle_coords = project_points_onto_line(Medial_ankle_coords,Midline_coords)
    length['Medial_ankle_length'] = calculate_distance(Medial_ankle_coords[0],Medial_ankle_coords[1])
    # print(Medial_ankle_length)

    #第五趾端点长度
    Five_toe_coords = mesh.points[coordinates['Five_toe_coords']]#[8471,12309]]
    Five_toe_coords = project_points_onto_plane(Five_toe_coords,foot_plane)
    Five_toe_coords = project_points_onto_line(Five_toe_coords,Midline_coords)
    length['Five_toe_length'] = calculate_distance(Five_toe_coords[0],Five_toe_coords[1])
    # print(Five_toe_length)

    #第五跖趾外凸点长度
    Five_toe_convex_coords = mesh.points[coordinates['Five_toe_convex_coords']]#[757,12309]]
    Five_toe_convex_coords = project_points_onto_plane(Five_toe_convex_coords,foot_plane)
    Five_toe_convex_coords = project_points_onto_line(Five_toe_convex_coords,Midline_coords)
    length['Five_toe_convex_length'] = calculate_distance(Five_toe_convex_coords[0],Five_toe_convex_coords[1])
    # print(Five_toe_convex_length)
    return length


def calculate_width(mesh,foot_plane,Midline_coords):
    width ={}
    #斜宽
    Oblique_width_coords = mesh.points[coordinates['Oblique_width_coords']]#[1717,1776]]
    Oblique_width_coords = project_points_onto_plane(Oblique_width_coords,foot_plane)
    width['Oblique_width_width'] = calculate_distance(Oblique_width_coords[0],Oblique_width_coords[1])
    # print(Oblique_width_width)

    #腰窝里段宽度
    Waist_dimples_coords = mesh.points[coordinates['Waist_dimples_coords']]#43]
    Waist_dimples_project_coords = project_point_onto_plane(Waist_dimples_coords,foot_plane)
    waist_dimples_projected_on_line = project_point_onto_line(Waist_dimples_project_coords,Midline_coords[0],Midline_coords[1])
    width['Waist_dimples_width'] = calculate_distance(Waist_dimples_project_coords,waist_dimples_projected_on_line)
    # print(Waist_dimples_width)

    # 踵心宽度
    Heel_width_coords = mesh.points[coordinates['Heel_width_coords']]#[3100,496]]
    Heel_width_coords = project_points_onto_plane(Heel_width_coords,foot_plane)
    width['Heel_width_width'] = calculate_distance(Heel_width_coords[0],Heel_width_coords[1])
    # print(Heel_width_width)

    #脚趾宽度
    toe_width_coords = mesh.points[coordinates['toe_width_coords']]#[757,451]]
    toe_width_coords = project_points_onto_plane(toe_width_coords,foot_plane)
    width['toe_width_width'] = calculate_distance(toe_width_coords[0],toe_width_coords[1])
    # print(toe_width_width)

    #第一跖趾里段宽度
    first_metatarsal_coords = mesh.points[coordinates['first_metatarsal_coords']]#1776]
    first_metatarsal_project_coords = project_point_onto_plane(first_metatarsal_coords,foot_plane)
    first_metatarsal_projected_coords= project_point_onto_line(first_metatarsal_project_coords,Midline_coords[0],Midline_coords[1])
    width['first_metatarsal_width']= calculate_distance(first_metatarsal_project_coords,first_metatarsal_projected_coords)
    # print(first_metatarsal_width)

    #第五跖趾里段宽度
    five_metatarsal_coords = mesh.points[coordinates['five_metatarsal_coords']]#1717]
    five_metatarsal_project_coords = project_point_onto_plane(five_metatarsal_coords,foot_plane)
    five_metatarsal_projected_coords= project_point_onto_line(five_metatarsal_project_coords,Midline_coords[0],Midline_coords[1])
    width['five_metatarsal_width']= calculate_distance(five_metatarsal_project_coords,five_metatarsal_projected_coords)
    # print(five_metatarsal_width)

    #第五趾外段宽度
    five_toe_coords = mesh.points[coordinates['five_toe_coords']]#757]
    five_toe_project_coords = project_point_onto_plane(five_toe_coords,foot_plane)
    five_toe_projected_coords= project_point_onto_line(five_toe_project_coords,Midline_coords[0],Midline_coords[1])
    width['five_toe_width']= calculate_distance(five_toe_project_coords,five_toe_projected_coords)
    # print(five_toe_width)

    #拇趾里段宽度
    thumb_inside_coords = mesh.points[coordinates['thumb_inside_coords']]#451]
    thumb_inside_project_coords = project_point_onto_plane(thumb_inside_coords,foot_plane)
    thumb_inside_projected_coords= project_point_onto_line(thumb_inside_project_coords,Midline_coords[0],Midline_coords[1])
    width['thumb_inside_width']= calculate_distance(thumb_inside_project_coords,thumb_inside_projected_coords)
    # print(thumb_inside_width)

    return width

def calculate_height(mesh,foot_plane):
    height = {}
    #拇指高度
    thumb_height_coords = mesh.points[coordinates['thumb_height_coords']]#[1106,1434]]
    height['thumb_height'] = calculate_distance(thumb_height_coords[0],thumb_height_coords[1])

    #后跟突点高度
    Heel_height_coords = mesh.points[coordinates['Heel_height_coords']]#12309]
    height['Heel_height'] = point_to_plane_distance(Heel_height_coords,foot_plane)

    #前跗骨高度
    Tarsal_height_coords = mesh.points[coordinates['Tarsal_height_coords']]#4993]
    height['Tarsal_height'] = point_to_plane_distance(Tarsal_height_coords,foot_plane)

    #外踝骨高度
    Outer_ankle = mesh.points[coordinates['Outer_ankle']]#3100]
    height['Outer_ankle_height'] = point_to_plane_distance(Outer_ankle,foot_plane)

    #内踝骨高度
    Inside_ankle_coords = mesh.points[coordinates['Inside_ankle_coords']]#496]
    height['Inside_ankle_height'] = point_to_plane_distance(Inside_ankle_coords,foot_plane)
    return height

def calculate_circumference(mesh):
    circumference = {}
    #计算脚趾围长
    Toe_circumference_coords = mesh.points[[451,727,757]]
    Toe_circumference__plane = calculate_plane_coeffs(Toe_circumference_coords[0],Toe_circumference_coords[1],Toe_circumference_coords[2])
    circumference['Toe_circumference'] = calculate_intersection_length(mesh, Toe_circumference__plane) 

    #跗趾围长
    Tarsal_circumference_coords = mesh.points[[1776,727,1717]]
    Tarsal_circumference__plane = calculate_plane_coeffs(Tarsal_circumference_coords[0],Tarsal_circumference_coords[1],Tarsal_circumference_coords[2])
    circumference['Tarsal_circumference'] = calculate_intersection_length(mesh, Tarsal_circumference__plane) 

    #兜跟围长
    Pocket_heel_circumference_coords = mesh.points[[8,671,3100]]
    Pocket_heel_circumference__plane = calculate_plane_coeffs(Pocket_heel_circumference_coords[0],Pocket_heel_circumference_coords[1],Pocket_heel_circumference_coords[2])
    circumference['Pocket_heel_circumference'] = calculate_intersection_length(mesh, Pocket_heel_circumference__plane) 

    #踝围围长
    Ankle_circumference_coords = mesh.points[[496,3100,671]]
    Ankle_circumference__plane = calculate_plane_coeffs(Ankle_circumference_coords[0],Ankle_circumference_coords[1],Ankle_circumference_coords[2])
    circumference['Ankle_circumference'] = calculate_intersection_length(mesh, Ankle_circumference__plane) 

    return circumference

def Calculating_model_parameters(length,pred_mesh_path):
        
    # 加载三维模型
    pred_mesh = pv.read(pred_mesh_path)

    # 选取三个顶点的坐标，用于构建脚底平面
    vertices_coords = pred_mesh.points[[1475, 3322, 28456]]

    # 创建脚底平面
    foot_plane = create_plane(vertices_coords[0], vertices_coords[1], vertices_coords[2])
    #计算脚长
    foot_bottom_length_coords = pred_mesh.points[[2504,42683,12309]]
    foot_bottom_length_coords = project_points_onto_plane(foot_bottom_length_coords,foot_plane)
    foot_bottom_length1 = calculate_distance(foot_bottom_length_coords[0],foot_bottom_length_coords[2])
    foot_bottom_length2 = calculate_distance(foot_bottom_length_coords[1],foot_bottom_length_coords[2])
    foot_length = max(foot_bottom_length1,foot_bottom_length2)
    
    #计算放大比例
    scale_factor = length / foot_length
    scaled_pred_mesh_points = pred_mesh.points * scale_factor
    pred_mesh.points = scaled_pred_mesh_points

    #计算中线
    Midline_coords = pred_mesh.points[[42683,12309]]

    #计算长度信息
    lengths = calculate_length(pred_mesh, foot_plane, Midline_coords)
    #计算宽度信息
    widths = calculate_width(pred_mesh, foot_plane, Midline_coords)
    #计算高度信息
    heights = calculate_height(pred_mesh, foot_plane)

    # circumference = calculate_circumference(pred_mesh)
    return lengths,widths,heights