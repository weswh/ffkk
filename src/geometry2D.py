import json
import os
import cv2
import math
import numpy as np
from pathlib import Path
from scipy.interpolate import splprep, splev, interp2d
# from src import read_write_colmap
import src.geometry3D as g3d
import open3d as o3d

def rename_images_in_folder(folder_path):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # 按自然排序文件，确保按顺序重命名
    files.sort()
    
    # 计数器初始化
    count = 1

    for file_name in files:
        # 获取文件的完整路径
        old_file_path = os.path.join(folder_path, file_name)
        
        # 确认文件是图像格式（以jpg或其他格式结尾）
        if file_name.lower().endswith(('jpg', 'jpeg', 'png')):
            # 新文件名，确保编号为三位数
            new_file_name = f"{str(count).zfill(3)}.jpg"
            new_file_path = os.path.join(folder_path, new_file_name)
            
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {old_file_path} to {new_file_path}")
            
            # 计数器递增
            count += 1

#def select_feet_masks(images, min_contour_area=10000, min_valid_masks=5, threshold_value=5):
# 筛选脚的mask
def select_feet_masks(images, min_contour_area=10000, min_valid_masks=5):
    n_images = len(images)
    h, w = images[0].shape[:2]
    masks = np.zeros((n_images, h, w), dtype=np.uint8)
    indices_valid = []
    for i in range(n_images):
        img = images[i]
        if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mk, _ = cv2.threshold(gray, 5, 255, 0)
        else:
             mk = img.copy().astype(np.uint8)
        #判断掩模图像是否提取到脚（面积大于10000）
        if len(np.unique(mk)) > 1:
            contours, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_contours = []
            for c in contours:
                if cv2.contourArea(c) > min_contour_area:
                    new_contours.append(c)
            if len(new_contours) != 0:
                 cv2.fillPoly(masks[i], new_contours, 255)
                 indices_valid.append(i)
    #检查是否提取足够多的图像
    if len(indices_valid) < min_valid_masks:
        raise ValueError("拍摄的图像质量不佳，请检查图像")

    return indices_valid, masks

# 去背景
def rgbs_masks_to_ngp(images, masks):
    n_images, h, w = images.shape[:3]
    ngp_images = np.zeros((n_images, h, w, 4), dtype=np.uint8)

    i = 0
    for img, mk in zip(images, masks):
        m_img = cv2.bitwise_and(img, img, mask=mk)
        rgba = cv2.cvtColor(m_img, cv2.COLOR_BGR2BGRA)
        rgba[np.where(mk==0)] = (0, 0, 0, 0)
        ngp_images[i] = rgba
        i += 1

    return ngp_images

# 已标定的广角相机去畸变
def undist_wide_cameras(images_path, transforms_path):
    #读取原图像，将去过畸变的图像存在u_images文件夹里
    images = sorted(os.listdir(images_path))
    u_images_path = os.path.join(os.path.dirname(images_path), "u_images")
    os.makedirs(u_images_path, exist_ok=True)
    with open(transforms_path, "r") as f:
        data = json.load(f) 
    #读取相机内参及畸变信息  
    for frame in data["frames"]:
        # image_name = Path(frame["file_path"]).name[:3]+".jpg"
        image_name = Path(frame["file_path"]).name#+".jpg"
        image_name = image_name .replace('.png','.jpg')
        # print(image_name)
        if image_name in images:
            image_path = os.path.join(images_path, image_name)
            image = cv2.imread(image_path)

            dist_coeffs = np.array(frame["dist"])
            intrinsic = np.eye(3)
            intrinsic[0][0] = frame["fl_x"]
            intrinsic[1][1] = frame["fl_y"]
            intrinsic[0][2] = frame["cx"]
            intrinsic[1][2] = frame["cy"]
            undistorted_image = cv2.undistort(image, intrinsic, dist_coeffs)
            u_image_path = os.path.join(u_images_path, image_name)
            cv2.imwrite(u_image_path, undistorted_image)
        else:
            continue

# mesh重投影验证
def reproject_mesh(mesh, images_path, transforms_path, outputs_name="eval_mesh"):
    outputs_path = os.path.join(os.path.dirname(images_path), outputs_name)
    os.makedirs(outputs_path, exist_ok=True)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    with open(transforms_path, "r") as f:
        data = json.load(f)

    for frame in data["frames"]:
        name = Path(frame["file_path"]).name.split(".")[0]
        image_path = os.path.join(images_path, name+".jpg")
        output_path = os.path.join(outputs_path, name+".jpg")

        c2w = np.array(frame["transform_matrix"])
        c2w[:, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        rvec, _ = cv2.Rodrigues(w2c[:3, :3])
        tvec = w2c[:3, 3:].reshape(-1, 3)

        K = np.eye(3)
        K[0][0], K[1][1], K[0][2], K[1][2] = frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]

        ps2d_pro, _ = cv2.projectPoints(vertices, rvec, tvec, K, np.zeros(4))
        ps2d_pro = np.int32(np.round(ps2d_pro)).reshape(-1, 2)

        image = cv2.imread(image_path)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for t in ps2d_pro[triangles]:
            cv2.fillConvexPoly(mask, t, 255)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0,0,255), 2)
        cv2.imwrite(output_path, image)

# mask投影验证
def eval_mask(masks_path, images_path):
    outputs_path = os.path.join(os.path.dirname(images_path), "eval_mask")
    os.makedirs(outputs_path, exist_ok=True)

    names = sorted(os.listdir(masks_path))
    for name in names:
        name = name.split(".")[0]
        mask_path = os.path.join(masks_path, name+".png")
        image_path = os.path.join(images_path, name+".jpg")
        output_path = os.path.join(outputs_path, name+".jpg")

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(output_path, image)

# 正交投影图像，重新描绘边界保存
def render_contour(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 200, 583, cv2.THRESH_BINARY_INV)  # 阈值化
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (153, 204, 255), thickness=cv2.FILLED)  # 填充轮廓
    cv2.imwrite(os.path.join(os.path.dirname(image_path), "obj2.png"), mask)

# 获取图像轮廓，并输出最大轮廓，只针对于单个足部
def detect_contour(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image 
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)  # 找到面积最大的轮廓
    return contour

# 点云去掉z值，投影到图像上
# pcd的x和y值，与image的x和y值方向相反，图像输出是从脚底板的投影点，以pcd为基准
def pcd_to_image(pcd, crop_z=None):
    pcd_t = g3d.crop_mesh(pcd, -1, crop_z)
    points = np.asarray(pcd_t.points)

    x_coords = points[:, 0]  # 提取 x 和 y 坐标
    y_coords = points[:, 1]

    image_width, image_height = 500, 500
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    for x, y in zip(x_coords+250, y_coords+250):
        cv2.circle(image, (int(x), int(y)), 1, (153, 204, 255), -1)

    contour = detect_contour(image)
    cv2.drawContours(image, [contour], -1, (153, 204, 255), thickness=cv2.FILLED)
    
    return image

# 图像中的足部基于contour转正
def forward_image(image):
    height, width = image.shape[0], image.shape[1]
    contour = detect_contour(image)
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    if angle <= 45:
        rotation_matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0) # left
    else:
        rotation_matrix = cv2.getRotationMatrix2D(center, float(angle-90), 1.0)  # right 正——逆时针旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return angle, center, rotated_image

# 转正求出点后，再把点转回去
def back_point(angle, center, point):
    if angle <= 45:
        inverse_angle = -angle
    else:
        inverse_angle = -(angle - 90)
    inverse_rotation_matrix = cv2.getRotationMatrix2D(center, inverse_angle, 1.0)  # 逆旋转矩阵

    point_homogeneous = np.array([point[0], point[1], 1]).reshape((3, 1))
    inverse_rotation_matrix_3x3 = np.vstack([inverse_rotation_matrix, [0, 0, 1]])  # 扩展为3x3矩阵
    original_point = np.dot(inverse_rotation_matrix_3x3, point_homogeneous).flatten()

    return original_point

# 找脚长百分比的二维点
def get_percentage_point2d(pcd, length, percentage):
    img = pcd_to_image(pcd, crop_z=20)  # 投影图像
    angle, center, image = forward_image(img)  # 转正图像
    contour = detect_contour(image)

    topmost = tuple(contour[contour[:,:,1].argmin()][0])  # 脚后跟
    given = length * (percentage / 100)
    given_y = topmost[1] + given
    given_point = (topmost[0], int(given_y))

    original_topmost = back_point(angle, center, topmost)
    original_given_point = back_point(angle, center, given_point)

    return original_topmost, original_given_point

# 在图像计算水平线、垂直线与contour的交点
def specified_line_contour_intersection(contour, point, line_type):
    intersections = []
    contour = np.asarray(contour).squeeze()

    if line_type == "horizen":  # 水平线
        y = point[1]
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) %len(contour)]
            if (p1[1] < y and p2[1] >= y) or (p1[1] >= y and p2[1] < y):
                x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                intersections.append((int(x), y))

    elif line_type == "vertical":  # 垂直线
        x = point[0]
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) %len(contour)]
            if (p1[0] < x and p2[0] >= x) or (p1[0] >= x and p2[0] < x):
                y = p1[1] + (x - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])
                intersections.append((x, int(y)))
    
    return intersections

# 计算图像中两个线段之间的夹角
def angle_between_lines(line1, line2):
    vec1 = np.array([line1[1][0] - line1[0][0], line1[1][1] - line1[0][1]])  # 计算向量
    vec2 = np.array([line2[1][0] - line2[0][0], line2[1][1] - line2[0][1]])

    norm_vec1 = np.linalg.norm(vec1)  # 计算向量的模（长度）
    norm_vec2 = np.linalg.norm(vec2)    

    dot_product = np.dot(vec1, vec2)  # 计算点积
    cos_theta = dot_product / (norm_vec1 * norm_vec2)  # 计算夹角的余弦值

    angle_rad = np.arccos(cos_theta)  # 计算夹角（弧度）
    angle_deg = np.degrees(angle_rad)  # 将夹角转换为角度
    
    return angle_deg

# 正交图像分成左右脚
# 这里默认的投影图是左边是左脚，右边是右脚
def left_right_image(image_path):
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # print('##########',image_path)
    height, width = image.shape[:2]

    center_x = width // 2
    center_y = height // 2

    # 左右分
    left_half = image[:, :center_x]
    right_half = image[:, center_x:]

    _, _, left_forward = forward_image(left_half)
    _, _, right_forward = forward_image(right_half)

    return left_forward, right_forward

# 输出转正图像的4个极端点
def detect_four_keypoints(image):
    contour = detect_contour(image)
    leftmost = tuple(contour[contour[:,:,0].argmin()][0])
    rightmost = tuple(contour[contour[:,:,0].argmax()][0])
    topmost = tuple(contour[contour[:,:,1].argmin()][0])
    bottommost = tuple(contour[contour[:,:,1].argmax()][0])

    center_x = (leftmost[0] + rightmost[0]) / 2
    center_y = (leftmost[1] + rightmost[1]) / 2
    center_point = (int(center_x), int(center_y))

    return leftmost, rightmost, topmost, bottommost, center_point

# 计算3个点的平面法向
def calculate_plane_normal(p1, p2, p3):
    p1 = np.array(p1)  # 将点转换为 NumPy 数组
    p2 = np.array(p2)
    p3 = np.array(p3)

    v1 = p2 - p1  # 计算向量
    v2 = p3 - p1

    normal = np.cross(v1, v2)  # 计算叉积
    normal = normal / np.linalg.norm(normal)  # 归一化

    return normal

# 21号图像输出到报告中
def draw_report_image(workfile):
    image_path = os.path.join(workfile, "u_image", "021.png")
    output_path = os.path.join(workfile, "feet_mod.png")

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    merged_contour = np.vstack(contours)
    M = cv2.moments(merged_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = (cX, cY)

    new_image = np.zeros_like(image)
    new_center = (image.shape[1] // 2, image.shape[0] // 2)
    dx = new_center[0] - center[0]
    dy = new_center[1] - center[1]

    non_alpha_channels = image[:, :, :3]
    non_zero_coords = np.argwhere(cv2.cvtColor(non_alpha_channels, cv2.COLOR_BGR2GRAY) > 0)
    for coord in non_zero_coords:
        y, x = coord
        new_y = y + dy - 150
        new_x = x + dx
        if 0 <= new_y < new_image.shape[0] and 0 <= new_x < new_image.shape[1]:
            new_image[new_y, new_x, :3] = image[y, x, :3]
            new_image[new_y, new_x, 3] = image[y, x, 3]

    cv2.imwrite(output_path, new_image)

def Detection_of_Hallux_Valgus(inputimg,name):
    img = inputimg
    # 查找最大轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    s_max = 0
    rect_max = 0
    contour_max = []
    for contour in contours:
        rect = cv2.minAreaRect(contour) # [中心点坐标，高度宽度，角度]
        s = rect[1][0]*rect[1][1]
        if s > s_max:
            s_max  = s
            rect_max = rect
            contour_max = contour

    h = []
    w = []
    if rect[1][0] < rect[1][1]:
        h =  rect[1][1]
        w =  rect[1][0]
    else:
        h =  rect[1][0]
        w =  rect[1][1]           
                
    # 脚后跟中点A      
    point_A = (np.intp(rect[0][0]),np.intp(rect[0][1]+h/2))
    
    # 求脚掌部分最大值第一跖趾外侧点E
    h_start = np.intp(rect[0][1]-h/2)
    h_end = np.intp(h_start + h/3)
    line = []
    id = []
    for i in range(h_start,h_end):
        row_pixels = img[i, :]
        # print(type(row_pixels))
        # print(row_pixels)
        non_zero_line = np.nonzero(row_pixels)
        line_mid = non_zero_line[0].tolist()
        if len(line_mid)!=0:
            # print(len(line_mid))
            if name == "r_img":# 右脚
                start_index = line_mid[0]
                end_index = np.intp(rect[0][0])
                line_mid  = end_index-start_index
                # print(line_mid)
                line.append(line_mid)
                le = (start_index,i)
                ri = (end_index,i)
                id.append((le,ri))
            
            if name == "l_img":# 左脚
                start_index = line_mid[-1]
                end_index = np.intp(rect[0][0])
                line_mid  = start_index-end_index
                # print(line_mid)
                line.append(line_mid)
                le = (end_index,i)
                ri = (start_index,i)
                id.append((le,ri))   
            
    line_max = 0
    id_max = []
    point_max = []
    for l in range(len(line)):
        if line[l]>line_max:
            line_max = line[l]
            id_max=  id[l]
            if name == "l_img":# 左脚
                point_max = id[l][1]
            if name == "r_img":# 右脚
                point_max = id[l][0]
    point_E = point_max
    
    find_tangent_contour = []
    if name == "l_img":# 左脚
        for contour_point in contour_max:
            if  (contour_point[0][0]>=point_A[0]) and (contour_point[0][1]<=point_E[1]):
                find_tangent_contour.append(contour_point)
    if name == "r_img":# 右脚
        for contour_point in contour_max:
            if  (contour_point[0][0]<=point_A[0]) and (contour_point[0][1]<=point_E[1]):
                find_tangent_contour.append(contour_point)
    
    point_x =[]
    point_y =[]
    for contour_point in find_tangent_contour:
        point_x.append(contour_point[0][0])
        point_y.append(contour_point[0][1])
    # print(point_x)
    # print(point_y)
    
    
    # 方法一
    #
    # 一条直线为垂直线，计算AE与OA夹角
    if point_A[0]-rect[0][0]==0:
        m1 = (point_A[1]-point_E[1])/(point_A[0]-point_E[0])
        angle_radians = math.atan(abs(1/m1))
        
    elif (point_A[0]-rect[0][0])!=0:
        m1 = (point_A[1]-point_E[1])/(point_A[0]-point_E[0])
        m2 = (point_A[1]-rect[0][1])/(point_A[0]-rect[0][0])
        
        # 计算夹角（弧度）
        angle_radians = math.atan(abs((m2 - m1) / (1 + m1 * m2)))
    print(angle_radians)
    # 转换为角度
    angle_degrees = math.degrees(angle_radians)
  
    # print(f"两直线之间的夹角为 {angle_degrees:.2f} 度")
    
    # if angle_degrees > 17:
    #     print("拇外翻")
    return angle_radians

#计算扁平足
def compute_flatfoot(path):
    img = input_pressure_img(path)
    #分开左右脚
    r_img,l_img = img2RL(img)
    #检测脚型
    left_type = Detection_of_flat_feet(l_img,"l_img")
    right_type = Detection_of_flat_feet(r_img,"r_img")
    
    return left_type, right_type
    
#计算拇外翻
def compute_Hallux_valgus(path):
    img = input_pressure_img(path)
    #分开左右脚
    r_img,l_img = img2RL(img)
    left_angle = Detection_of_Hallux_Valgus(r_img,"r_img")
    right_angle = Detection_of_Hallux_Valgus(l_img,"l_img")
    return left_angle,right_angle


def input_pressure_img(path):
    ## 导入72*80图像扩展为720*800并模糊化,用于计算
    gray_img = cv2.imread(path,0)
    gray_img = cv2.resize(gray_img, (800, 720),interpolation=cv2.INTER_LINEAR) # 将72行80列的压力板图像扩展10倍interpolation=cv2.INTER_CUBIC
    img = gray_img
    # 高斯滤波
    img = cv2.GaussianBlur(img, (5, 5), 0) #可以更改核大小
    ret,img=cv2.threshold(img, 10, 255, cv2.THRESH_TOZERO)
    return img

def Detection_of_flat_feet(input_img,name):
    img = input_img
    # 查找最大轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 查找最大轮廓的最小外接矩形
    s_max, rect_max, center = 0, None, None
    for contour in contours:
        rect = cv2.minAreaRect(contour) # [中心点坐标，高度宽度，角度]
        area = rect[1][0] * rect[1][1]
        if area > s_max:
            s_max = area
            center = rect[0]
            rect_max = rect
            h = max(rect[1])
    min_id, max_id = int(center[1] - h / 6), int(center[1] + h / 6)
    # 计算足弓的最小宽度值
    line_min, id_min = _find_min_width(img, min_id, max_id)

    # 计算脚掌的最大宽度值
    line_max, id_max = _find_max_width(img, max_id)

    # 评估足部情况
    flatfoot_ratio = line_max / line_min
    if flatfoot_ratio >= 2:
        foot_type = 'normal'
    elif 1.5 < flatfoot_ratio < 2:
        foot_type = 'mild'
    elif 1 < flatfoot_ratio <= 1.5:
        foot_type = 'moderate'
    else:
        foot_type = 'severe'
    
    return foot_type

def _find_min_width(img, min_id, max_id):
    """
    计算给定范围内的最小宽度值。
    """
    line_min, id_min = 9999, []
    for i in range(min_id, max_id):
        row_pixels = img[i, :]
        non_zero_indices = np.nonzero(row_pixels)[0].tolist()
        if non_zero_indices:
            width = non_zero_indices[-1] - non_zero_indices[0]
            if width < line_min:
                line_min = width
                id_min = [(non_zero_indices[0], i), (non_zero_indices[-1], i)]
    return line_min, id_min

def _find_max_width(img, max_id):
    """
    计算给定范围内的最大宽度值。
    """
    line_max, id_max = 0, []
    for i in range(max_id):
        row_pixels = img[i, :]
        non_zero_indices = np.nonzero(row_pixels)[0].tolist()
        if non_zero_indices:
            width = non_zero_indices[-1] - non_zero_indices[0]
            if width > line_max:
                line_max = width
                id_max = [(non_zero_indices[0], i), (non_zero_indices[-1], i)]
    return line_max, id_max

def img2RL(input_img):
    """
    将图像分割为左右两脚，并对每个脚图像进行旋转校正。
    """
    img = input_img
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找出最大的两个轮廓，用于表示左右脚
    r_rect, l_rect = _find_two_largest_rectangles(contours)
    
    # 分割图像为左右脚
    center = ((r_rect[0][0] + l_rect[0][0]) * 0.5, (r_rect[0][1] + l_rect[0][1]) * 0.5)
    center = np.intp(center)
    l_img, r_img = img[:, :center[0]], img[:, center[0] + 1:]
    
    # 旋转左右脚图像回正
    l_img = _rotate_image_to_upright(l_img)
    r_img = _rotate_image_to_upright(r_img)
    
    return r_img, l_img

def _find_two_largest_rectangles(contours):
    """
    找出轮廓中的两个最大矩形。
    """
    rl_s = [0, 0]
    r_rect, l_rect = None, None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        area = rect[1][0] * rect[1][1]
        if area > max(rl_s):
            if rl_s[0] > rl_s[1]:
                rl_s[1], l_rect = rl_s[0], r_rect
            rl_s[0], r_rect = area, rect
        elif area > rl_s[1]:
            rl_s[1], l_rect = area, rect
    return r_rect, l_rect

def _rotate_image_to_upright(img):
    """
    旋转图像使其回正。
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=lambda cnt: cv2.contourArea(cnt))
    rect = cv2.minAreaRect(largest_contour)
    
    angle = rect[2]
    if rect[1][0] < rect[1][1]:
        angle = -angle if angle > 0 else angle
    else:
        angle = 90 - angle if angle > 0 else 90 + angle

    rotation_matrix = cv2.getRotationMatrix2D(center=rect[0], angle=angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    
    return rotated_img
# 
# def filter_mesh_by_y_coordinate(mesh):
#     # 获取顶点坐标
#     vertices = np.asarray(mesh.vertices)
#     vertices_to_keep = vertices[:, 0] <= 0
#     triangles_to_remove = np.any(vertices_to_keep[mesh.triangles], axis=1)
#     mesh.remove_triangles_by_mask(triangles_to_remove)
#     mesh.remove_unreferenced_vertices()
#     mesh.remove_degenerate_triangles()
#     return mesh