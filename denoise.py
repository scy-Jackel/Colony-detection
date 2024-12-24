import cv2
import numpy as np
import os


def process_mask_image(input_dir, output_dir):
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录下的所有jpg文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            # 构建输入文件路径和输出文件路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 读取掩码图像 (假设图像是单通道灰度图像)
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"无法读取文件 {filename}, 跳过...")
                continue

            # 使用霍夫圆变换检测圆形
            circles = cv2.HoughCircles(
                img,
                cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                param1=50, param2=30, minRadius=4500, maxRadius=6000
            )

            if circles is not None:
                # 选择检测到的第一个圆形
                circles = np.round(circles[0, :]).astype("int")
                x, y, r = circles[0]  # 获取圆形的中心 (x, y) 和半径 r

                # 创建一个全白的图像
                mask = np.ones_like(img) * 255

                # 在全白图像上绘制黑色圆形区域，表示圆形内的部分
                cv2.circle(mask, (x, y), r - 200, (0, 0, 0), -1)

                # 使用按位OR将圆形区域与原图进行合成，圆形外的区域置为白色
                result = cv2.bitwise_or(img, mask)

                # 将结果图像保存到输出目录
                cv2.imwrite(output_path, result)
                print(f"处理并保存图像：{filename}")
            else:
                print(f"未找到圆形：{filename}")


if __name__ == "__main__":
    input_dir = './seg_result'  # 输入目录
    output_dir = './denoise'  # 输出目录
    process_mask_image(input_dir, output_dir)
