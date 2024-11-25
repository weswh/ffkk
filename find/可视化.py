import pyvista as pv
import numpy as np

# 加载三维模型
mesh = pv.read('/media/liuyalan/Projects/足扫项目/FOOT/FIND/eval_export/eval_3d/3D_only/FIND/meshes/00_pred_mesh.ply')
# mesh = pv.read('/media/liuyalan/Projects/足扫项目/FIND/data/Meshes_sliced-20240926T014445Z-001/Meshes_sliced/0003/A/0003-A.obj')
# 获取顶点
# points = mesh.points
# # 创建可视化窗口
# plotter = pv.Plotter()

# # 添加三维模型

# plotter.add_mesh(mesh, color='lightblue')

# # 为每个顶点添加索引标签
# indices = np.arange(points.shape[0])  # 顶点索引
# plotter.add_point_labels(points, indices, point_size=10, font_size=12)

# # 显示模型
# plotter.show()


# 获取顶点
points = mesh.points

# 创建可视化窗口
plotter = pv.Plotter()

# 添加三维模型
plotter.add_mesh(mesh, color='lightblue')

# 指定要显示索引的顶点列表（例如，显示索引为 0, 10, 50 的顶点）
selected_indices =  [1106,1434,13651,157,1469,496,421,3100,1717,4993,43,1776,1967,451,727,429,8,2868,1246,757,6465,1391,2504,42683, 2221,671,3208 , 3727,  8471, 12309,1475,3322,28456]
# selected_indices = [ 2504,42683,  3208 , 3727,  8471, 12309]

# 仅提取指定顶点的位置和索引
selected_points = points[selected_indices]
selected_labels = np.array(selected_indices)

# 为指定的顶点添加索引标签
plotter.add_point_labels(selected_points, selected_labels, point_size=10, font_size=12)

# 显示模型
plotter.show()