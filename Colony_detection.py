import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure
from cellpose import models

# 输入和输出文件夹路径
input_folder = r'./data'
output_folder = r'./cellpose'

# 如果输出文件夹不存在，创建一个
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化 Cellpose 模型，使用 'cyto' 模型
model = models.Cellpose(gpu=True, model_type='nuclei')

# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # 根据需要可以添加其他图像格式
        # 构建图像路径
        image_path = os.path.join(input_folder, filename)

        # 读取图像
        img = io.imread(image_path)

        # 使用 Cellpose 进行细胞分割
        masks, flows, styles, diams = model.eval(img, diameter=30,
                                                 flow_threshold=0.5,  # 增大以提高边界精度
                                                 cellprob_threshold=-3.0,  # 降低以检测更多细胞
                                                 channels=[0, 0])

        # 使用连通域分析识别菌落
        labeled_bacteria = measure.label(masks)

        # 保存原图（无需修改）
        # original_image_path = os.path.join(output_folder, f'{filename}_original.png')
        # io.imsave(original_image_path, img)

        # 保存细胞分割图像
        # segmented_cells_path = os.path.join(output_folder, f'{filename}_segmented_cells.png')
        # io.imsave(segmented_cells_path, masks.astype(np.uint16))

        # 保存菌落分割图像
        detected_colonies_path = os.path.join(output_folder, f'{filename}_detected_colonies.png')
        io.imsave(detected_colonies_path, labeled_bacteria.astype(np.uint16))

        # 创建合成图像：原始图像和菌落分割图像叠加
        # combined_image = np.zeros_like(img)  # 创建一个与原图相同大小的空白图像
        #
        # # 将原始图像叠加到合成图像上
        # combined_image = np.dstack([img, img, img])  # 假设原图是灰度图，将其转换为3通道图像以便叠加
        #
        # # 将菌落分割图像叠加到合成图像上
        # combined_image = combined_image * 0.5 + np.dstack([labeled_bacteria, labeled_bacteria, labeled_bacteria]) * 0.5  # 调整透明度
        #
        # # 保存合成图像
        # combined_image_path = os.path.join(output_folder, f'{filename}_combined_image.png')
        # io.imsave(combined_image_path, (combined_image).astype(np.uint8))

        print(f"Processed {filename} and saved results to {output_folder}")