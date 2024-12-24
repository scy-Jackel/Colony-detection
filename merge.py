import cv2
import numpy as np
import os

# 输入图像文件夹路径
original_images_path = r'./data'
mask_images_path = r'./denoise'

# 输出图像文件夹路径
output_path = r'./merge'

# 获取文件夹中所有图像文件
original_images = os.listdir(original_images_path)
mask_images = os.listdir(mask_images_path)

# 确保输出文件夹存在
os.makedirs(output_path, exist_ok=True)

# 遍历所有图像
for original_image_name in original_images:
    # 确保掩膜图像与原图名称相同（假设文件名完全一致）
    mask_image_name = original_image_name

    # 获取原图和掩膜图像的完整路径
    original_image_path_full = os.path.join(original_images_path, original_image_name)
    mask_image_path_full = os.path.join(mask_images_path, mask_image_name)

    # 读取原图和掩膜图像
    original_image = cv2.imread(original_image_path_full)
    mask_image = cv2.imread(mask_image_path_full, cv2.IMREAD_GRAYSCALE)

    if original_image is None or mask_image is None:
        print(f"无法读取图像：{original_image_name} 或掩膜图像：{mask_image_name}")
        continue

    # 确保两张图像尺寸一致
    if original_image.shape[:2] != mask_image.shape:
        print(f"图像尺寸不一致：{original_image_name} 和 {mask_image_name}，跳过")
        continue

    # 创建一个与原始图像相同大小的红色图层
    red_layer = np.zeros_like(original_image, dtype=np.uint8)
    red_layer[:, :, 2] = 255  # 红色通道最大值

    # 创建掩膜：黑色部分设置为红色区域
    alpha_mask = (mask_image == 0).astype(np.uint8)  # 黑色部分的掩膜
    alpha_mask = cv2.merge([alpha_mask, alpha_mask, alpha_mask])  # 转换为三通道掩膜

    # 将红色图层与原始图像按一定透明度叠加
    alpha = 0.5  # 半透明强度
    fused_image = cv2.addWeighted(original_image, 1.0, red_layer, alpha, 0)

    # 使用掩膜将红色区域替换为原始图像中的黑色区域
    fused_image = np.where(alpha_mask == 0, original_image, fused_image)  # 将非黑色部分恢复为原始图像

    # 输出融合后的图像路径
    output_image_path = os.path.join(output_path, original_image_name)

    # 保存融合后的图像
    cv2.imwrite(output_image_path, fused_image)
    print(f"融合后的图像已保存到：{output_image_path}")
