# import os
# import numpy as np
# from PIL import Image

# # 文件夹路径
# folder_path = '/home/veily/gaussian_surfels/data/FEET_TEST/20240517/cy/01/stable/'

# # 获取文件夹中所有以 .png 结尾的文件
# file_list = sorted([file for file in os.listdir(folder_path) if file.endswith('.png')])

# # 循环处理每个文件
# for idx, file_name in enumerate(file_list):
#     # 构造文件的编号，从 001 开始的三位数
#     file_idx = f"{idx+1:03}"
#     # 构造完整的文件路径
#     file_path = os.path.join(folder_path, file_name)
    
#     # 读取 numpy 文件
#     img_np =Image.open(file_path)
    
#     # 将所有像素值取负
#     #img_np_neg = -img_np
    
#     # 打印信息或进行进一步处理（这里仅打印示例）
#     #print(f"Processed file {file_name}, shape: {img_np.shape}, min value: {np.min(img_np_neg)}, max value: {np.max(img_np_neg)}")
#     #img_np = np.array(img_np)
#     img_np = np.array(img_np, dtype=np.float32)
#     img_np[:, :, :3] *= -1

#     # 将 numpy 数组保存到文件
#     save_file_path = os.path.join(folder_path, f"{file_idx}_normal.npy")
#     #print(img_np/255.0+1)
#     np.save(save_file_path, img_np/255.0+1)
#     img_neg = Image.fromarray(np.uint8(img_np))
    
#     # 保存图像文件
#     save_img_path = os.path.join(folder_path, f"{file_idx}_negative.png")
#     img_neg.save(save_img_path)
#     #print(f"Saved numpy array (negated) to {save_file_path}")
import os
import numpy as np
from PIL import Image

# 文件夹路径
folder_path = '/home/veily/gaussian_surfels/data/FEET_TEST/20240517/cy/01/stable/'


# 获取文件夹中所有以 .png 结尾的文件，并按文件名中的数字顺序排序
file_list = sorted([file for file in os.listdir(folder_path) if file.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0]))

# 循环处理每个文件
for idx, file_name in enumerate(file_list):
    # 构造文件的编号，从 001 开始的三位数
    file_idx = f"{idx+1:03}"
    # 构造完整的文件路径
    file_path = os.path.join(folder_path, file_name)
    
    # 读取图片文件
    img = Image.open(file_path)
    
    # 转换为 numpy 数组，并转换数据类型为 float32
    img_np = np.array(img, dtype=np.float32)
    
    # 将第二维和第三维的元素取负
    img_np[:, :, :3] *= -1
    save_file_path = os.path.join(folder_path, f"{file_idx}_normal.npy")
    print(img_np/255.0+1)
    np.save(save_file_path, img_np/255.0)
    #img_neg = Image.fromarray(np.uint8(img_np))
    # 转换回 PIL 图像
    img_neg = Image.fromarray(np.uint8(img_np))
    
    # 构造保存图像文件名，保持原始文件名顺序
    save_img_path = os.path.join(folder_path, f"{file_idx}_negative.png")
    
    # 保存图像文件
    img_neg.save(save_img_path)
    print(f"Saved image (negative) to {save_img_path}")
