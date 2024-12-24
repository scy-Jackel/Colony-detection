import cv2
import numpy as np
import os

# 输入图像文件夹路径
input_folder = r'./cellpose'  # 输入文件夹，存放原始图像
output_folder = r'./seg_cellpose'  # 输出文件夹，保存处理后的图像

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有图像文件
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# 遍历所有图像文件
for image_name in image_files:
    # 获取图像的完整路径
    image_path = os.path.join(input_folder, image_name)

    # 读取图像
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像：{image_name}")
        continue

    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将黑色区域变为白色，非黑色区域变为黑色
    binary_image = np.where(gray_image == 0, 255, 0).astype(np.uint8)  # 黑色变白，其他变黑

    # 将二值图像转换为三通道图像，以便保存为彩色图像
    binary_image_colored = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # 修改文件名：去掉 ".jpg_detected_colonies"
    output_image_name = image_name.replace('.jpg_detected_colonies', '')  # 去除指定的字符串

    # 输出处理后的图像路径
    output_image_path = os.path.join(output_folder, output_image_name)

    # 保存处理后的图像
    cv2.imwrite(output_image_path, binary_image_colored)
    print(f"处理后的图像已保存到：{output_image_path}")