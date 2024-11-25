# #! /usr/bin/python3
# # -*- coding: utf-8 -*-
# import math
# import time
# import numpy as np
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline

# ## 计算脚掌面积
# def The_area_of_the_soles(input_img): #--输入原始压力图
#     # 确定灰度大于 0 的像素个数
#     non_zero_count = np.count_nonzero(input_img) # 有压力的感应点个数
#     s = non_zero_count*25 # 单个感应点为5mm*5mm
#     return s # 返回面积


# ## 计算足压峰值  （问题：目前未确定压力值与输出值之间的对应关系，无法计算具体压力大小）
# def Peak_foot_pressure(input_img): #--输入单通道灰度压力图
#     # 找到最大灰度值及其位置
#     (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(input_img)

#     print(f"最大灰度值: {maxVal}")
#     print(f"最大灰度值所在的位置: {maxLoc}")

#     # 显示原图像并标记出最大灰度值的位置
#     # marked_image = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
#     # cv2.circle(marked_image, maxLoc, 10, 100, 2)
#     # cv2.imshow('Max Gray Pixel', marked_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
    
#     Peak_foot = maxVal * 0.01 # 传感器感应点电压
#     return Peak_foot, maxLoc # 返回位点电压和位点坐标


# # 计算平均足压
# def Mean_foot_pressure(input_img): #--输入单通道灰度压力图
#     # 找到所有灰度值大于零的像素
#     non_zero_pixels = input_img[input_img > 0]
#     # 计算这些像素的平均值
#     if len(non_zero_pixels) > 0:
#         average_value = np.mean(non_zero_pixels)
#         print(f"所有灰度值大于零的像素的平均值是: {average_value}")
#         return average_value*0.01
#     else:
#         print("没有灰度值大于零的像素")
#         return 0

# # 单独计算左右脚脚掌脚跟压力
# def Mean_soles_heels_pressure(input_img): #--输入单通道灰度压力图
#     img = input_img
#     height, width = img.shape
#     # 计算分割点（图像宽度的一半）
#     mid_point1 = width // 2
#     mid_point2 = height // 2
#     # 分割图像为两部分
#     left_up_image = img[:mid_point2-2, :mid_point1]  # 左上部分
#     left_down_image = img[mid_point2+2:, :mid_point1]  # 左下部分
#     right_up_image = img[:mid_point2-2, mid_point1:]  # 右上部分
#     right_down_image = img[mid_point2+2:, mid_point1:]  # 右下部分
    
#     l_u = Mean_foot_pressure(left_up_image)
#     l_d = Mean_foot_pressure(left_down_image)
#     r_u = Mean_foot_pressure(right_up_image)
#     r_d = Mean_foot_pressure(right_down_image)
    
#     print("left_up_image",l_u)
#     print("left_down_image",l_d)
#     print("right_up_image",r_u)
#     print("right_down_image",r_d)
#     return 0


# ## 显示动态足压力 21-30为一组
# def Changes_in_foot_pressure(img_list_l,img_list_r): # --输入序列图片
#     pressure_list_l = []
#     pressure_list_r = []
#     for img in img_list_l:
#         Peak_foot, maxLoc = Peak_foot_pressure(img)
#         pressure_list_l.append(Peak_foot)
#     for img in img_list_r:
#         Peak_foot, maxLoc = Peak_foot_pressure(img)
#         pressure_list_r.append(Peak_foot)
    
#     num = len(pressure_list_l)
#     # 设置字体为黑体，确保支持中文显示
#     plt.clf()
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
#     plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#     # 创建折线图
#     plt.plot(range(0,num), pressure_list_l, marker='o')  # 使用 'o' 来标记每个点
#     plt.plot(range(0,num), pressure_list_r, marker='*')  # 使用 'o' 来标记每个点
#     # 添加标题和标签
#     plt.title('动态足压力')
#     plt.xlabel('序号')
#     plt.ylabel('压力板电压')
#     # 显示图形
#     plt.grid(True)  # 添加网格线使图形更清晰
#     plt.legend()  # 添加图例
#     # 保存图形到文件
#     plt.savefig('./output/foot_pressure_comparison.png', dpi=300)  # dpi 参数设置图像清晰度
#     plt.show()
#     return 0


# ## 显示动态图像分左右脚 21-30为一组
# def The_list_of_pressure_image(img_list):
#     left_list = []
#     right_list =[]
#     # 获取图像的尺寸
#     for path in img_list:
#         img = cv2.imread(path,0)
#         height, width = img.shape
#         # 计算分割点（图像宽度的一半）
#         mid_point = width // 2
#         # 分割图像为两部分
#         left_image = img[:, :mid_point]  # 左半部分
#         right_image = img[:, mid_point:]  # 右半部分
#         left_list.append(left_image)
#         right_list.append(right_image)
        
#     Changes_in_foot_pressure(left_list,right_list) # 绘制折线图
    
#     # 使用第一张图像的大小作为基准
#     height, width = left_list[0].shape
#     resized_images_l = [cv2.resize(img, (width, height)) for img in left_list]
#     horizontal_concat_l = np.hstack(resized_images_l)
#     height, width = right_list[0].shape
#     resized_images_r = [cv2.resize(img, (width, height)) for img in right_list]
#     horizontal_concat_r = np.hstack(resized_images_r)
    
#     l_img = show_heatmap(horizontal_concat_l)
#     r_img = show_heatmap(horizontal_concat_r)

#     # 显示拼接后的图片
#     cv2.imshow('l', l_img)
#     cv2.imwrite('./output/l.jpg', l_img)
#     cv2.imshow('r', r_img)
#     cv2.imwrite('./output/r.jpg', r_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return horizontal_concat_l,horizontal_concat_r


# ##--------------------------------------------------------------------------------------------
# ##--------------------------------------------------------------------------------------------
# ## 将图片转换为黑底的热力图图像
# def show_heatmap(inputimg):
#      # 创建一个黑色背景的图像
#     black_background = np.zeros_like(inputimg)
#     # 仅将非零像素复制到黑色背景
#     non_zero_pixels = inputimg > 0
#     black_background[non_zero_pixels] = inputimg[non_zero_pixels]
#     # 将处理后的图像应用伪彩色映射
#     heatmap = cv2.applyColorMap(black_background, cv2.COLORMAP_JET)
#     # 将黑色背景的图像和热力图合并
#     final_heatmap = np.zeros_like(heatmap)
#     final_heatmap[non_zero_pixels] = heatmap[non_zero_pixels]
#     return final_heatmap
# ##--------------------------------------------------------------------------------------------
# ##--------------------------------------------------------------------------------------------



# def compute_pressure_param(path):
#     # path = "C:\\Users\\19785\\Desktop\\zyh\\code\\zs\\web_img\\20241011_22\\pressure_img.png"
#     img = cv2.imread(path,0)
#     Mean_soles_heels_pressure(img)
    
#     # # 显示动态足压力 21-30为一组
#     # imglist = []
#     # root_path = "C:\\Users\\19785\\Desktop\\zyh\\code\\zs\\web_img\\20241011_"
#     # for i in range(21,31):
#     #     imgpath = root_path + str(i) + "\\pressure_img.png"
#     #     imglist.append(imgpath)
#     # Changes_in_foot_pressure(imglist)
#     # print(imglist[1])
    
#     # # 显示动态足压力 21-30为一组
#     # imglist = []
#     # root_path = "C:\\Users\\19785\\Desktop\\zyh\\code\\zs\\web_img\\20241011_"
#     # for i in range(21,30):
#     #     imgpath = root_path + str(i) + "\\pressure_img.png"
#     #     imglist.append(imgpath)
#     # The_list_of_pressure_image(imglist)
    
#     print("over")













import math
import time
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import json
import cv2
import numpy as np
import src.file_io as fio
import os
## 计算脚掌面积
def The_area_of_the_soles(input_img): #--输入原始压力图
    # 确定灰度大于 0 的像素个数
    non_zero_count = np.count_nonzero(input_img) # 有压力的感应点个数
    s = non_zero_count*25 # 单个感应点为5mm*5mm
    return s # 返回面积

## 显示动态图像分左右脚 21-30为一组
def The_list_of_pressure_image(img_list):
    left_list = []
    right_list =[]
    # 获取图像的尺寸
    for path in img_list:
        img = cv2.imread(path,0)
        height, width = img.shape
        # 计算分割点（图像宽度的一半）
        mid_point = width // 2
        # 分割图像为两部分
        left_image = img[:, :mid_point]  # 左半部分
        right_image = img[:, mid_point:]  # 右半部分
        left_list.append(left_image)
        right_list.append(right_image)
        
    Changes_in_foot_pressure(left_list,right_list) # 绘制折线图
    
    # 使用第一张图像的大小作为基准
    height, width = left_list[0].shape
    resized_images_l = [cv2.resize(img, (width, height)) for img in left_list]
    horizontal_concat_l = np.hstack(resized_images_l)
    height, width = right_list[0].shape
    resized_images_r = [cv2.resize(img, (width, height)) for img in right_list]
    horizontal_concat_r = np.hstack(resized_images_r)
    
    l_img = show_heatmap(horizontal_concat_l)
    r_img = show_heatmap(horizontal_concat_r)

    # 显示拼接后的图片
    cv2.imshow('l', l_img)
    cv2.imwrite('./output/l.jpg', l_img)
    cv2.imshow('r', r_img)
    cv2.imwrite('./output/r.jpg', r_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return horizontal_concat_l,horizontal_concat_r

def show_heatmap(inputimg):
     # 创建一个黑色背景的图像
    black_background = np.zeros_like(inputimg)
    # 仅将非零像素复制到黑色背景
    non_zero_pixels = inputimg > 0
    black_background[non_zero_pixels] = inputimg[non_zero_pixels]
    # 将处理后的图像应用伪彩色映射
    heatmap = cv2.applyColorMap(black_background, cv2.COLORMAP_JET)
    # 将黑色背景的图像和热力图合并
    final_heatmap = np.zeros_like(heatmap)
    final_heatmap[non_zero_pixels] = heatmap[non_zero_pixels]
    return final_heatmap

def Peak_foot_pressure(input_img):
    # Find the maximum gray value and its location
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(input_img)
    Peak_foot = maxVal * 0.01  # Convert the max gray value to foot pressure
    return Peak_foot, maxLoc  # Return peak pressure and its location

def Mean_foot_pressure(input_img):
    # Calculate the mean pressure of non-zero pixels
    non_zero_pixels = input_img[input_img > 0]
    if len(non_zero_pixels) > 0:
        average_value = np.mean(non_zero_pixels)
        return average_value * 0.01  # Convert mean gray value to foot pressure
    else:
        return 0  # Return 0 if no pressure values are found

def Mean_soles_heels_pressure(img):
    # Split the input image into four quadrants
    height, width = img.shape
    mid_point = width // 2
    left_up_image = img[:height//2, :mid_point]  # Left upper quadrant
    left_down_image = img[height//2:, :mid_point]  # Left lower quadrant
    right_up_image = img[:height//2, mid_point:]  # Right upper quadrant
    right_down_image = img[height//2:, mid_point:]  # Right lower quadrant
    
    # Get mean pressures for each quadrant
    results = {
        "left_up": Mean_foot_pressure(left_up_image),
        "left_down": Mean_foot_pressure(left_down_image),
        "right_up": Mean_foot_pressure(right_up_image),
        "right_down": Mean_foot_pressure(right_down_image)
    }
    return results  # Return a dictionary of results

def save_results_to_json(results, data_path, filename='results.json'):
    # Construct the full file path
    full_path = f"{data_path}/{filename}"
    # Save results to the specified JSON file
    with open(full_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)  # Indent for readability

## 显示动态足压力 21-30为一组
def Changes_in_foot_pressure(img_list_l,img_list_r): # --输入序列图片
    pressure_list_l = []
    pressure_list_r = []
    for img in img_list_l:
        Peak_foot, maxLoc = Peak_foot_pressure(img)
        pressure_list_l.append(Peak_foot)
    for img in img_list_r:
        Peak_foot, maxLoc = Peak_foot_pressure(img)
        pressure_list_r.append(Peak_foot)
    
    num = len(pressure_list_l)
    # 设置字体为黑体，确保支持中文显示
    plt.clf()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 创建折线图
    plt.plot(range(0,num), pressure_list_l, marker='o')  # 使用 'o' 来标记每个点
    plt.plot(range(0,num), pressure_list_r, marker='*')  # 使用 'o' 来标记每个点
    # 添加标题和标签
    plt.title('动态足压力')
    plt.xlabel('序号')
    plt.ylabel('压力板电压')
    # 显示图形
    plt.grid(True)  # 添加网格线使图形更清晰
    plt.legend()  # 添加图例
    # 保存图形到文件
    plt.savefig('./output/foot_pressure_comparison.png', dpi=300)  # dpi 参数设置图像清晰度
    plt.show()
    return 0

def split_image(img):
    # Get the dimensions of the image
    height, width = img.shape
    mid_point = width // 2  # Calculate the midpoint

    # Split the image into left and right halves
    left_image = img[:, :mid_point]  # Left half
    right_image = img[:, mid_point:]  # Right half

    return left_image, right_image
# Example usage
def compute_pressure_param(path,data_path):
    # Load the pressure image
    img = cv2.imread(path, 0)
    left_img, right_img = split_image(img)  # Split the image
    # Calculate mean pressures for the soles and heels
    pressure_results = Mean_soles_heels_pressure(img)
    left_Max_Peak_foot = Peak_foot_pressure(left_img)
    right_Max_Peak_foot = Peak_foot_pressure(right_img)
    left_feet_area = The_area_of_the_soles(left_img)
    right_feet_area = The_area_of_the_soles(right_img)
    left_mean_pressure = Mean_foot_pressure(left_img)
    right_mean_pressure = Mean_foot_pressure(left_img)
    ret = { "left": {"Max_Peak_foot":left_Max_Peak_foot,
                            "feet_area": left_feet_area, 
                            "mean_pressure": left_mean_pressure 
                    },
            "right":{ "Max_Peak_foot": right_Max_Peak_foot ,
                            "feet_area": right_feet_area, 
                            "mean_pressure":  right_mean_pressure
                    }}
    # Save results to JSON
    fio.save_json(os.path.join(data_path, "results.json"), ret)
    # # # 显示动态足压力 21-30为一组
    # imglist = []
    # for i in range(21,31):
    #     imgpath = data_path + str(i) + "\\pressure_img.png"
    #     imglist.append(imgpath)
    # Changes_in_foot_pressure(imglist)
    # print(imglist[1])
    
    # # 显示动态足压力 21-30为一组
    # imglist = []
    # for i in range(21,30):
    #     imgpath = data_path + str(i) + "\\pressure_img.png"
    #     imglist.append(imgpath)
    # The_list_of_pressure_image(imglist)
    # print("Results saved to results.json")  # Confirmation message