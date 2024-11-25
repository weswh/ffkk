
import cv2
import numpy as np
import json
import open3d as o3d
import os
def process_frame(frame_data, image_path, ply_path):
    # 解析帧数据
    fl_x = frame_data['fl_x']
    fl_y = frame_data['fl_y']
    cx = frame_data['cx']
    cy = frame_data['cy']
    #dist_coeffs = frame_data['dist']
    dist_coeffs = None
    transform_matrix = np.array(frame_data['transform_matrix'], dtype=np.float64)

    # 反转变换矩阵
    matrix=transform_matrix
    matrix[0:3, 1:3] *= -1
    matrix = np.linalg.inv(matrix)

    # 加载图像
    image = cv2.imread(image_path)

    # 设置相机参数
    camera_matrix = np.array([[fl_x, 0, cx],
                              [0, fl_y, cy],
                              [0, 0, 1]], dtype=np.float64)

    #dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
    dist_coeffs=None
    # 读取PLY网格模型
    mesh = o3d.io.read_triangle_mesh(ply_path)

    # 获取网格顶点坐标
    vertices = np.asarray(mesh.vertices)
    translation_vector=np.array([0,-0.7,0])
    translated_vertices = vertices + translation_vector
    mesh.vertices =o3d.utility.Vector3dVector(translated_vertices)
    vertices = np.asarray(mesh.vertices)
    # rotation_matrix = np.array([
    #     [0,0,1],
    #     [0,1,0],
    #     [-1,0,0]
    # ])
    # rotated_vertices=np.dot(vertices,rotation_matrix.T)
    # mesh.vertices =o3d.utility.Vector3dVector(rotated_vertices)
    vertices = np.asarray(mesh.vertices)
    # 应用变换矩阵
    transformed_vertices = np.matmul(vertices, matrix[:3, :3].T) + matrix[:3, 3]

    # 投影网格模型到图像
    image_points, _ = cv2.projectPoints(transformed_vertices, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
    image_points = np.int32(image_points).reshape(-1, 2)

    # 创建掩码并绘制轮廓
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # for face in mesh.triangles:
    #     pts = np.array([image_points[face[0]], image_points[face[1]], image_points[face[2]]], np.int32)
    #     pts = pts.reshape((-1, 1, 2))
    #     cv2.fillConvexPoly(mask, pts, 255)
    
    if len(mesh.triangles) > 0:
        for face in mesh.triangles:
            pts = np.array([image_points[face[0]], image_points[face[1]], image_points[face[2]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillConvexPoly(mask, pts, 255)
    else:
        # 如果没有三角形信息，则直接绘制点云
        for pt in image_points:
            cv2.circle(mask, tuple(pt), 1, 255, -1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 3)
    save_path = os.path.splitext(image_path)[0] + '_processed.jpg'
    cv2.imwrite(save_path, image)
    return image

def main():
    # 读取JSON文件
    json_path = '/media/liuyalan/Projects/足扫项目/gaussian_surfels/0909/202409090005/transforms.json'
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 循环处理每个帧
    for frame_data in data['frames']:
        # 构建图像路径和PLY路径
        image_path = '/media/liuyalan/Projects/足扫项目/gaussian_surfels/0909/202409090005/u_images/' + frame_data['file_path'].split('/')[-1]
        image_path = image_path.replace('png','jpg')
        print(image_path)
        ply_path = '/media/liuyalan/Projects/足扫项目/gaussian_surfels/0909/202409090005/mesh.ply'
       
        #ply_path = '/home/veily/gaussian_surfels/output/FEET_TEST/20240527/01/09_2c8b5559-1/point_cloud/iteration_5000/point_cloud.ply'
        # 处理当前帧数据
        result_image = process_frame(frame_data, image_path, ply_path)
        
        # 显示处理结果
        #cv2.imshow('Projected Model', result_image)
        #cv2.waitKey(0)

    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
