import os
import cv2
import numpy as np
import open3d as o3d
import src.geometry2D as g2d
import src.geometry3D as g3d
import math
import src.pressure_output as pro


# 检测足弓二维点
def detect_arch2d(pcd, direction, length):
    pts = np.asarray(pcd.points)
    # 找到最低点（heel）和最高点（fingertip），即足跟和脚尖的 y 坐标
    heel = pts[np.argmin(pts[:, 1])][1]
    fingertip = pts[np.argmax(pts[:, 1])][1]
    # 计算足跟和脚尖之间六分之一的距离
    ranges = length / 6
    # 将三维点云投影为二维图像
    im = g2d.pcd_to_image(pcd, crop_z=13)
    image = np.zeros_like(im)
    # 根据足跟和脚尖的 y 坐标范围筛选图像区域
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if (heel+ranges+250) <= y <= (fingertip-ranges+250):
                image[y, x, :] = im[y, x, :]

    contour = g2d.detect_contour(image)
    hull = cv2.convexHull(contour)
    hulls = np.asarray(hull).squeeze()
    # 获取凸包点索引，用于检测凸缺陷
    hull_point = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull_point)
    # 初始化变量，用于存储最大凸缺陷点（max_concave）及其距离（max_concave_dist）
    max_concave = None
    max_concave_dist = -1
    # 遍历所有凸缺陷，找到距离最大的缺陷点
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        far = tuple(contour[f][0])
        if d > max_concave_dist:
            max_concave = far
            max_concave_dist = d
    # 计算水平线与轮廓的交点
    intersections = g2d.specified_line_contour_intersection(hulls, max_concave, line_type="horizen")  # 求水平线与contour的交点
    points = np.asarray(intersections)
    # 根据方向找到最左侧或最右侧的交点
    if direction == "left": indice = np.argmax(points[:, 0])
    elif direction == "right": indice = np.argmin(points[:, 0])
    point = points[indice]

    # mask = np.zeros_like(image)
    mask = image.copy()
    cv2.drawContours(mask, contour, -1, (153, 204, 255), thickness=cv2.FILLED)  # 填充轮廓
    cv2.drawContours(mask, [hull], -1, (0, 255, 0), 2)     # 绘制凸包
    cv2.circle(mask, max_concave, 5, (255, 0, 0), -1)  # 绘制最大凸包点
    cv2.line(mask, (0, max_concave[1]), (mask.shape[1], max_concave[1]), (0, 255, 0), 2)  # 过凸点绘制水平线
    cv2.circle(mask, point, 5, (0, 0, 255), -1)  # 绘制凸包轮廓与水平线的相交点

    return mask, point

## 检测拇指外翻
def detect_hallux_valgus(pcd, direction, length):
    pts = np.asarray(pcd.points)
    # 找到最高点，即脚尖的 y 坐标
    fingertip = pts[np.argmax(pts[:, 1])][1]
    # 沿 y 轴方向裁剪点云，只保留脚尖区域
    ranges = length / 3
    pcd_cropped = g3d.crop_mesh_axis(pcd, fingertip-ranges, fingertip, "y")
    img_cropped = g2d.pcd_to_image(pcd_cropped, crop_z=20)
    rot_img = cv2.rotate(img_cropped, cv2.ROTATE_180)
    angle, center, for_img = g2d.forward_image(rot_img)  # 转正图像
    left, right, _, _, center = g2d.detect_four_keypoints(for_img)

    image = g2d.pcd_to_image(pcd, crop_z=20)
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    # 计算旋转矩阵，使脚尖方向向上
    if angle <= 45:
        rotation_matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0) # left
    else:
        rotation_matrix = cv2.getRotationMatrix2D(center, float(angle-90), 1.0)  # right 正——逆时针旋转
    for_image = cv2.warpAffine(rotated_image, rotation_matrix, (image.shape[0], image.shape[1]))
    _, _, top, bottom, _ = g2d.detect_four_keypoints(for_image)

    # mask = np.zeros_like(rotated_image)
    mask = for_image.copy()
    contour = g2d.detect_contour(for_image)
    cv2.drawContours(mask, contour, -1, (153, 204, 255), thickness=cv2.FILLED)  # 填充轮廓
    cv2.circle(mask, left, 5, (0, 0, 255), -1)
    cv2.circle(mask, right, 5, (0, 255, 0), -1)
    cv2.circle(mask, top, 5, (255, 0, 0), -1)
    cv2.circle(mask, bottom, 5, (255, 255, 255), -1)
    cv2.circle(mask, center, 5, (0, 255, 255), -1)

    if direction == "left":
        line1 = [bottom, center]
        line2 = [bottom, left]
        cv2.line(mask, bottom, center, (0, 255, 0), 2)
        cv2.line(mask, bottom, left, (0, 255, 0), 2)

    elif direction == "right":
        line1 = [bottom, center]
        line2 = [bottom, right]
        cv2.line(mask, bottom, center, (0, 255, 0), 2)
        cv2.line(mask, bottom, right, (0, 255, 0), 2)

    angle = g2d.angle_between_lines(line1, line2)

    return mask, angle

# 小于18正常；18-25轻度；25-45中度；大于45重度
def analyse_hallux_valgus(angle):
    result = "neutral"
    if angle < 18: result = "neutral"
    elif angle >= 18 and angle < 25: result = "mild"
    elif angle >= 25 and angle < 45: result = "moderate"
    else: result = "serve"

    return result

## 力线角度检测
# 点云裁剪脚后跟和脚踝
def crop_heel_pcd(pcd, length):
    img = g2d.pcd_to_image(pcd, crop_z=20)
    angle, center, image = g2d.forward_image(img)
    contour = g2d.detect_contour(image)
    
    heel_point2d, front_heel_point2d = g2d.get_percentage_point2d(pcd, length=length, percentage=18)
    intersections_2d = g2d.specified_line_contour_intersection(contour, front_heel_point2d, line_type='horizen')
    intersections_2d1 = g2d.back_point(angle, center, intersections_2d[0])
    intersections_2d2 = g2d.back_point(angle, center, intersections_2d[1])

    # cv2.circle(img, (int(heel_point2d[0]), int(heel_point2d[1])), 1, (0, 255, 0), -1)
    # cv2.circle(img, (int(front_heel_point2d[0]), int(front_heel_point2d[1])), 1, (255, 255, 0), -1)
    # cv2.circle(img, (int(intersections_2d1[0]), int(intersections_2d1[1])), 1, (0, 255, 255), -1)
    # cv2.circle(img, (int(intersections_2d2[0]), int(intersections_2d2[1])), 1, (255, 0, 0), -1)
    # cv2.imwrite("/home/veily/Feet3D/data/20240718/01/202407180015/debug0.png", img)

    heel_point3d = g3d.get_point3d(pcd, heel_point2d, part="closest")
    # front_heel_point3d = g3d.get_point3d(pcd, front_heel_point2d, part="closest")
    front_heel_point3d = [front_heel_point2d[0]-250, front_heel_point2d[1]-250, heel_point3d[2]+50]
    intersections_3d1 = g3d.get_point3d(pcd, intersections_2d1, part="closest")
    intersections_3d2 = g3d.get_point3d(pcd, intersections_2d2, part="closest")

    plane_normal = g2d.calculate_plane_normal(front_heel_point3d, intersections_3d1, intersections_3d2)
    if plane_normal[1] < 0 :
        plane_normal[1] = -plane_normal[1]
    cropped_pcd = g3d.crop_pcd_with_plane(pcd, front_heel_point3d, plane_normal)
    ankle_pcd = g3d.crop_mesh(cropped_pcd, 50, 100)

    rotated_pcd, rotation_matrix= g3d.rotate_mesh_with_axis(cropped_pcd, -90, "x")
    rotated_ankle_pcd, _ = g3d.rotate_mesh_with_axis(ankle_pcd, -90, "x")
    heel_point3d_rotated = np.dot(rotation_matrix[:3, :3], np.array(heel_point3d))

    return heel_point3d_rotated, rotated_pcd, rotated_ankle_pcd

# 检测足后跟力线角度
def detect_heel_angle(pcd, length):
    heel_point3d, heel_pcd, ankle_pcd = crop_heel_pcd(pcd, length=length)

    ankle_img = g2d.pcd_to_image(ankle_pcd, np.max(np.asarray(ankle_pcd.points)[:, 2]))
    ankle_contour = g2d.detect_contour(ankle_img)

    rect = cv2.minAreaRect(ankle_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    center = np.mean(box, axis=0)

    img = g2d.pcd_to_image(heel_pcd, np.max(np.asarray(heel_pcd.points)[:, 2]))
    contour = g2d.detect_contour(img)
    heel_point2d = (heel_point3d[0]+250, heel_point3d[1]+250)
    intersections = g2d.specified_line_contour_intersection(contour, heel_point2d, line_type='vertical')
    heel_point = min(intersections, key=lambda point: point[1])

    line1 = [center, (center[0], center[1]-50)]
    line2 = [center, heel_point]
    angle = g2d.angle_between_lines(line1, line2)

    cv2.circle(img, (int(center[0]),int(center[1])), 2, (0, 0, 255), -1)
    cv2.circle(img, (int(heel_point[0]),int(heel_point[1])), 2, (255, 255, 255), -1)
    cv2.line(img, (int(center[0]),int(center[1])), (int(center[0]),int(center[1]-50)), (0, 255, 0), 1)
    cv2.line(img, (int(center[0]),int(center[1])), (int(heel_point[0]),int(heel_point[1])), (0, 255, 0), 1)

    return img, angle, center, heel_point

# 分析力线角度
# 0-8度正常；8-12度轻度；12度以上重度
def analyse_heel_inex(angle, center_point, heel_point, direction):
    center = center_point[0]
    heel = heel_point[0]
    result = "neutral"

    if direction == "left":
        if center > heel:
            if angle >= 8 and angle < 12: result = "in"
            elif angle >= 12: result = "serve in"
            else: result = "neutral"
        else:
            if angle >= 8 and angle < 12: result = "out"
            elif angle >= 12: result = "serve out"
            else: result = "neutral"

    elif direction == "right":
        if center < heel:
            if angle >= 8 and angle < 12: result = "in"
            elif angle >= 12: result = "serve in"
            else: result = "neutral"
        else:
            if angle >= 8 and angle < 12: result = "out"
            elif angle >= 12: result = "serve out"
            else: result = "neutral"
            
    return result

#检测足弓高，力线角度，拇外翻
def detect_foot_params(pcd, direction, length):
    pcd = g3d.forward_pcd(pcd)
    img = g2d.pcd_to_image(pcd, crop_z=20)
    angle, center, image = g2d.forward_image(img)
    
    arch_img, arch2d = detect_arch2d(pcd, direction=direction, length=length)
    arch3d = g3d.get_point3d(pcd, arch2d, part="closest")

    _, instep2d = g2d.get_percentage_point2d(pcd, length=length, percentage=50)
    instep3d = g3d.get_point3d(pcd, instep2d, part="y_zmax")

    hallux_img, hallux_angle = detect_hallux_valgus(pcd, direction=direction, length=length)
    hallux_result = analyse_hallux_valgus(hallux_angle)

    # heel_img, heel_angle, heel_center, heel_point = detect_heel_angle(pcd, length=length)
    # heel_result = analyse_heel_inex(heel_angle, heel_center, heel_point, direction=direction)

    result = {
        "arch_height": round(arch3d[2], 2), 
        "instep_height": round(instep3d[2], 2), 
        
        "hallux_angle":round(hallux_angle, 2), 
        "is_hallux_valgused": hallux_result, 

        # "heel_angle":round(heel_angle, 2),
        # "is_heel_valgused": heel_result
    }

    return arch_img, hallux_img, result

# 计算尺码
def calculate_shoe_size(length):
    size = round((length / 10 + 2) * 1.5, 1)

    if size - int(size) == 0.5:  # 如果小数点后一位是5，则保留，否则四舍五入为整数
        size = round(size, 1)  # 保留一位小数
    else:
        size = int(round(size, 0))  # 四舍五入为整数

    return size

def calculate_us_size(length):
    size = int(round((length / 10 - 18) + 0.5, 0))

    return size

# 计算重力中心
def caculate_non_blue_area(image):
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    # lower_blue = np.array([100, 50, 50])
    # upper_blue = np.array([140, 255, 255])

    # blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # non_blue_mask = cv2.bitwise_not(blue_mask)
    # non_blue_area = np.sum(non_blue_mask > 0)

    # return non_blue_area
    alpha = image[:, :, 3]
    non_alpha = np.count_nonzero(alpha)
    return non_alpha

# 计算重心偏移
def calculate_gravity(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    left_image, right_image = g2d.left_right_image(gray_image)
    left = caculate_non_blue_area(left_image)
    right = caculate_non_blue_area(right_image)

    total = left + right
    difference = np.abs(left - right)
    ratio = (difference / total) * 100

    gravity = "normal"
    if ratio > 8:
        if left > right:
            gravity = "left"
        else:
            gravity = "right"
    else:
        gravity = "normal"
    return gravity

# 扁平足判断
def check_flat(ratio):
    type = "normal"
    if ratio > 0.67: 
        type = "flat"
    elif ratio < 0.3: 
        type = "cavus"
    else: 
        type = "normal"
    return type
# 计算扁平足
def if_flatfooted(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]
    center_x = width // 2

    left = image[:, :center_x]
    right = image[:, center_x:]

    left_alpha = left[:, :, 3]
    right_alpha = right[:, :, 3]

    left_non_alpha = np.count_nonzero(left_alpha)
    right_non_alpha = np.count_nonzero(right_alpha)

    return left_non_alpha, right_non_alpha

#判断是否是拇外翻
def overlap_mesh_press(mesh, pressure_path):
    points = np.asarray(mesh.vertices)
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    repro = np.zeros((500, 500, 3), dtype=np.uint8)
    for x, y in zip(x_coords+250, y_coords+250):
        cv2.circle(repro, (int(x), int(y)), 1, (153, 204, 255), -1)

    press_point = np.array([[0, 720], [0, 0], [800, 0], [800, 720]], dtype=np.float32)  # pressure.jpg
    # image_point = np.array([[456-33,421-5], [454-33, 70-5], [63-33, 74-5], [61-33, 422-5]], dtype=np.float32)  # repro.jpg 
    #image_point = np.array([[456-20,421+4], [454-20, 70+4], [63-20, 74+4], [61-20, 422+4]], dtype=np.float32)  # repro.jpg 
    
    image_point = np.array([[456-24,421+8], [454-24, 70+8], [63-24, 74+8], [61-24, 422+8]], dtype=np.float32)
    H, _ = cv2.findHomography(image_point, press_point)
    image_wraped = cv2.warpPerspective(repro, H, (800, 720))
    gray = cv2.cvtColor(image_wraped, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    press = cv2.imread(pressure_path)
    press_with_contours = press.copy()
    cv2.drawContours(press_with_contours, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(os.path.dirname(pressure_path), "overlap.png"), press_with_contours)

    left, right = g2d.left_right_image(os.path.join(os.path.dirname(pressure_path), "overlap.png"))
    left_cnt = g2d.detect_contour(left)
    right_cnt = g2d.detect_contour(right)

    left_contour = cv2.contourArea(left_cnt)
    right_contour = cv2.contourArea(right_cnt)

    left_press, right_press = if_flatfooted(pressure_path)
    left_ratio = left_press / left_contour
    right_ratio = right_press / right_contour

    left_type = check_flat(left_ratio)
    right_type = check_flat(right_ratio)

    return left_type, right_type

def input_mesh_img(path):
    """
    读取并处理网格图像，应用高斯模糊和形态学操作。
    
    参数:
        path (str): 图像的文件路径。
        
    返回:
        img (numpy.ndarray): 处理后的图像。
    """
    # 读取灰度图像
    gray_img = cv2.imread(path, 0)
    
    # 应用高斯模糊
    img = cv2.GaussianBlur(gray_img, (3, 3), 0)  # 核大小可以根据需要调整

    # 形态学操作：闭合边缘
    kernel = np.ones((3, 3), np.uint8)  # 卷积核
    img = cv2.dilate(img, kernel, iterations=10)  # 连续膨胀操作
    img = cv2.erode(img, kernel, iterations=10)   # 连续腐蚀操作

    return img


def compute_hallux_valgus(path):
    """
    计算图像中左右脚的拇外翻角度及类型。
    
    参数:
        path (str): 图像的文件路径。
        
    返回:
        tuple: 左脚和右脚的拇外翻角度和类型 (左角度, 右角度, 左类型, 右类型)。
    """
    img = input_mesh_img(path)
    r_img, l_img = g2d.img2RL(img)
    
    # 检测左脚和右脚的拇外翻角度和类型
    right_angle, right_type = Detection_of_Hallux_Valgus(r_img, "r_img")
    left_angle, left_type = Detection_of_Hallux_Valgus(l_img, "l_img")
    
    return left_angle, right_angle, left_type, right_type

def Detection_of_Hallux_Valgus(input_img, img_name):
    """
    检测单脚的拇外翻角度和类型。
    
    参数:
        input_img (numpy.ndarray): 单脚图像。
        img_name (str): 图像名称（"r_img" 或 "l_img"）。
        
    返回:
        tuple: 拇外翻角度及类型 (角度, 类型)。
    """
    img = input_img
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大轮廓及其外接矩形
    max_area, max_rect, max_contour = 0, None, None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        area = rect[1][0] * rect[1][1]
        if area > max_area:
            max_area = area
            max_rect = rect
            max_contour = contour

    # 确定脚的宽度和高度
    width, height = sorted(max_rect[1], reverse=True)
    
    # 计算脚后跟中点A
    point_A = (int(max_rect[0][0]), int(max_rect[0][1] + height / 2))

    # 确定脚掌的第一跖趾外侧点E
    h_start = int(max_rect[0][1] - height / 2)
    h_end = int(h_start + height / 3)
    
    point_E = calculate_max_width_point(img, h_start, h_end, max_rect, img_name)
    
    # 计算脚后跟与拇趾连线的斜率，并与垂直线计算夹角
    angle_degrees = calculate_angle_between_lines(point_A, point_E, max_rect[0])

    # 分析拇外翻类型
    hallux_valgus_type = analyse_hallux_valgus(angle_degrees)
    
    return angle_degrees, hallux_valgus_type

def calculate_max_width_point(img, h_start, h_end, rect, img_name):
    """
    计算脚掌部分最大宽度的点E。
    
    参数:
        img (numpy.ndarray): 处理过的图像。
        h_start (int): 脚掌开始行。
        h_end (int): 脚掌结束行。
        rect (tuple): 最小外接矩形信息。
        img_name (str): 图像名称（"r_img" 或 "l_img"）。
        
    返回:
        tuple: 最大宽度的点E坐标。
    """
    max_width, point_E = 0, None
    
    for i in range(h_start, h_end):
        row_pixels = img[i, :]
        non_zero_indices = np.nonzero(row_pixels)[0].tolist()

        if non_zero_indices:
            if img_name == "r_img":  # 右脚
                width = int(rect[0][0]) - non_zero_indices[0]
                if width > max_width:
                    max_width = width
                    point_E = (non_zero_indices[0], i)
            else:  # 左脚
                width = non_zero_indices[-1] - int(rect[0][0])
                if width > max_width:
                    max_width = width
                    point_E = (non_zero_indices[-1], i)
    
    return point_E

def calculate_angle_between_lines(point_A, point_E, rect_center):
    """
    计算点A和点E连线与垂直线的夹角。
    
    参数:
        point_A (tuple): 脚后跟中点坐标。
        point_E (tuple): 第一跖趾外侧点坐标。
        rect_center (tuple): 矩形中心点坐标。
        
    返回:
        float: 计算得到的角度。
    """
    if point_A[0] == rect_center[0]:
        # 直线垂直
        slope_AE = (point_A[1] - point_E[1]) / (point_A[0] - point_E[0])
        angle_radians = math.atan(abs(1 / slope_AE))
    else:
        # 计算斜率并求夹角
        slope_AE = (point_A[1] - point_E[1]) / (point_A[0] - point_E[0])
        slope_OA = (point_A[1] - rect_center[1]) / (point_A[0] - rect_center[0])
        angle_radians = math.atan(abs((slope_OA - slope_AE) / (1 + slope_AE * slope_OA)))

    # 将角度转换为度
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees