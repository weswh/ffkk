# #! /usr/bin/python3
# # -*- coding: utf-8 -*-
# import math
# import time
# import numpy as np
# import cv2

# def find_hpoint(max_contour,box):
#     len_c = len(max_contour)
#     if len_c<5:
#         return 0
# #  # 计算交点
# #     for i in range(4):
# #         p1, p2 = tuple(box[i]), tuple(box[(i + 1) % 4])
# #         for j in range(len(max_contour)):
# #             p3, p4 = tuple(max_contour[j][0]), tuple(max_contour[(j + 1) % len(max_contour)][0])
# #             intersection = line.intersection(p1, p2, p3, p4)
# #             if intersection:
# #                 cv2.circle(max_contour, (int(intersection[0]), int(intersection[1])), 5, (255, 0, 0), -1)
#     # 直接遍历求轮廓最长点,取5次求平均
#     max_line = [0, 0, 0, 0, 0]
#     max_line_id = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
#     flag = 0
#     for t in range(0,5):
#         for i in range(0,len_c):
#             flag = 0
#             for j in range(1,len_c):
#                 x = max_contour[i][0]
#                 y = max_contour[j][0]
#                 line = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
                
#                 for k in range(0,t):
#                     if [i,j] == max_line_id[k] or [j,i] == max_line_id[k]:
#                         flag = 1
#                         # print("跳过")
#                         break
#                 if flag == 0:
#                     if line > max_line[t]:
#                         max_line[t] = line
#                         max_line_id[t][0] = i
#                         max_line_id[t][1] = j
#                 flag = 0         
#     return max_line

# def find_minbox(path):
#     gray_img  = cv2.imread("./0828/202408280008/obj2.png",0)
#     # print(gray_img)
#     img = gray_img
#     ret,img=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    
#     # 形态学操作连接
#     kernel = kernel = np.ones((3, 3), np.uint8) # 卷积核
#     for i in range(0,9):
#         img = cv2.dilate(img, kernel, iterations=1)# 膨胀
#     for i in range(0,9):
#         img = cv2.erode(img, kernel, iterations=1) # 腐蚀

#     # 查找轮廓
#     contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # print(len(contours))
#     # 轮廓遍历
#     scale = 0.5 #像素与实际距离比例
#     box_img = gray_img
#     for max_contour in contours:
#         rect = cv2.minAreaRect(max_contour)
#         if abs(rect[0][0]-rect[0][1])<30:
#             break
#         box = cv2.boxPoints(rect)
#         box = np.intp(box)
#         a = find_hpoint(max_contour,box)
#         print(a)
#         # print(box)
#         w = math.dist(box[0], box[1])*scale
#         h = math.dist(box[1], box[2])*scale
#         if(w>h):
#             a = w
#             w = h
#             h = a
#         print("-----------------------")
#         print("w:",str(w)+"mm")
#         print("h:",str(h)+"mm")
#         print("-----------------------")
#         # 绘制矩形
#         box_img = cv2.drawContours(box_img.copy(), [box], 0, 100, 2)
#         # cv2.imshow("output_img",box_img)
#         # cv2.waitKey (0) 
#         # cv2.destroyAllWindows()
#         # 保存结果
#         cv2.imwrite('./rect_image.png', box_img)

# if __name__ == "__main__":
#     find_minbox(12)

    
#!/usr/bin/python3
# -*- coding: utf-8 -*-



# import math
# import time
# import numpy as np
# import cv2
# import os

# def find_hpoint(max_contour, box):
#     len_c = len(max_contour)
#     if len_c < 5:
#         return 0

#     max_line = [0, 0, 0, 0, 0]
#     max_line_id = [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
#     flag = 0
#     for t in range(0, 5):
#         for i in range(0, len_c):
#             flag = 0
#             for j in range(1, len_c):
#                 x = max_contour[i][0]
#                 y = max_contour[j][0]
#                 line = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

#                 for k in range(0, t):
#                     if [i, j] == max_line_id[k] or [j, i] == max_line_id[k]:
#                         flag = 1
#                         break
#                 if flag == 0:
#                     if line > max_line[t]:
#                         max_line[t] = line
#                         max_line_id[t][0] = i
#                         max_line_id[t][1] = j
#                 flag = 0      
#     # print(':@@@@@@@@@@@',max_line)   
#     return max_line

# def find_minbox(path):
#     gray_img = cv2.imread(path, 0)
#     ret, img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
    
#     # 形态学操作连接
#     kernel = np.ones((3, 3), np.uint8)
#     for i in range(0, 11):
#         img = cv2.dilate(img, kernel, iterations=1)
#     for i in range(0, 11):
#         img = cv2.erode(img, kernel, iterations=1)

#     # 查找轮廓
#     contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     scale = 0.5  # 像素与实际距离比例
#     box_img = gray_img
#     for max_contour in contours:
#         rect = cv2.minAreaRect(max_contour)
#         if abs(rect[0][0] - rect[0][1]) < 40:
#             break
#         box = cv2.boxPoints(rect)
#         box = np.intp(box)
#         a = find_hpoint(max_contour, box)
#         if(a!=0):
#         #    print(a)
#             average_a = sum(a) / len(a)
#             print("Average of a:", average_a)
#         w = math.dist(box[0], box[1]) * scale
#         h = math.dist(box[1], box[2]) * scale
#         if w > h:
#             w, h = h, w
#         # print("-----------------------")
#         # print(f"w: {w} mm")
#         # print(f"h: {h} mm")
#         # print("-----------------------")
#         box_img = cv2.drawContours(box_img.copy(), [box], 0, 100, 2)

#         # 保存结果
#         result_path = path.replace("obj2.png", "rect_image.png")
#         # print(result_path)
#         cv2.imwrite(result_path, box_img)

# if __name__ == "__main__":
#     # 批量处理从202408280008到202408280088
#     for i in range(7, 7):
#         folder_name = f"20240828{i:04d}"
#         image_path = f"./0828/{folder_name}/obj2.png"
#         print(f"Processing {image_path}...")
#         find_minbox(image_path)


#! /usr/bin/python3
# -*- coding: utf-8 -*-
import math
import time
import numpy as np
import cv2

def find_hpoint(max_contour,box):
    len_c = len(max_contour)
    if len_c<5:
        return 0
#  # 计算交点
#     for i in range(4):
#         p1, p2 = tuple(box[i]), tuple(box[(i + 1) % 4])
#         for j in range(len(max_contour)):
#             p3, p4 = tuple(max_contour[j][0]), tuple(max_contour[(j + 1) % len(max_contour)][0])
#             intersection = line.intersection(p1, p2, p3, p4)
#             if intersection:
#                 cv2.circle(max_contour, (int(intersection[0]), int(intersection[1])), 5, (255, 0, 0), -1)
    # 直接遍历求轮廓最长点,取5次求平均
    max_line = [0, 0, 0, 0, 0]
    max_line_id = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
    flag = 0
    for t in range(0,5):
        for i in range(0,len_c):
            flag = 0
            for j in range(1,len_c):
                x = max_contour[i][0]
                y = max_contour[j][0]
                line = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
                
                for k in range(0,t):
                    if [i,j] == max_line_id[k] or [j,i] == max_line_id[k]:
                        flag = 1
                        # print("跳过")
                        break
                if flag == 0:
                    if line > max_line[t]:
                        max_line[t] = line
                        max_line_id[t][0] = i
                        max_line_id[t][1] = j
                flag = 0         
    return max_line, max_line_id

def find_minbox(path):
    gray_img  = cv2.imread(path,0)
    img = gray_img
    ret,img=cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    
    # 形态学操作连接
    kernel = kernel = np.ones((3, 3), np.uint8) # 卷积核
    for i in range(0,9):
        img = cv2.dilate(img, kernel, iterations=1)# 膨胀
        # cv2.imshow("output_img",img)
        # cv2.waitKey (0) 
        # cv2.destroyAllWindows()
    for i in range(0,9):
        img = cv2.erode(img, kernel, iterations=1) # 腐蚀
        # cv2.imshow("output_img",img)
        # cv2.waitKey (0) 
        # cv2.destroyAllWindows()

    # 查找轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # 轮廓遍历
    scale = 1 #像素与实际距离比例
    box_img = gray_img
    left=[]
    right=[]
    for max_contour in contours:
        rect = cv2.minAreaRect(max_contour)
        if abs(rect[0][0]-rect[0][1])<30:
            break
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        a,id = find_hpoint(max_contour,box)
        if a!=0 :
            box_data = a
            print(id)
            print(box_data)
        # print(box)
        w = math.dist(box[0], box[1])*scale
        h = math.dist(box[1], box[2])*scale
        if(w>h):
            a = w
            w = h
            h = a
        # print("-----------------------")
        # print("w:",str(w)+"mm")
        # print("h:",str(h)+"mm")
        # print("-----------------------")
        if(box[0][0]<250):
            left.append(w)
            left.append(h)
        else:
            right.append(w)
            right.append(h)
        # 绘制矩形
        box_img = cv2.drawContours(box_img.copy(), [box], 0, 100, 2)
        for ids in id:
            # print(ids[0])
            # print(max_contour[0][0][0])
            ptStart = max_contour[ids[0]][0]
            ptEnd = max_contour[ids[1]][0]
            
            ptStart = (ptStart[0],ptStart[1])
            ptEnd = (ptEnd[0],ptEnd[1])
            
            # print(ptStart)
            # print(ptEnd)
            point_color = 255 # BGR
            thickness = 1 
            lineType = 4
            cv2.line(box_img, ptStart, ptEnd, point_color, thickness, lineType)

        # cv2.imshow("output_img",box_img)
        # cv2.waitKey (0) 
        # cv2.destroyAllWindows()
        # 保存结果
    image_path = path.replace('obj2.png','rect_image.png')
    cv2.imwrite(image_path, box_img)
    return left[0],left[1],right[0],right[1]
if __name__ == "__main__":
    find_minbox(12)