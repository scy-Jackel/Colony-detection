import cv2
import numpy as np
import os

# 图像处理函数
def process_image(image):
    # 第一段：处理为灰度并将大于70的灰度值设置为255
    processed_image = image.copy()
    processed_image[processed_image > 70] = 255

    # 第二段：滑动窗口（窗口大小11），均值小于150则设置为黑色
    h, w = processed_image.shape
    window_size = 11
    half_window = window_size // 2
    temp_image = np.full_like(processed_image, 255)  # 初始化为白色图像
    for y in range(half_window, h - half_window):
        print(1, y)
        for x in range(half_window, w - half_window):
            window = processed_image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            window_mean = np.mean(window)
            if window_mean < 150:
                temp_image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1] = 0
    processed_image = temp_image.copy()

    # 第三段：滑动窗口（窗口大小50），均值小于60则设置为黑色
    window_size = 50
    half_window = window_size // 2
    temp_image = np.full_like(processed_image, 255)  # 初始化为白色图像
    for y in range(half_window, h - half_window):
        print(2, y)
        for x in range(half_window, w - half_window):
            window = processed_image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            window_mean = np.mean(window)
            if window_mean < 60:
                temp_image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1] = 0
    processed_image = temp_image.copy()

    # 第四段：找到连通区域并移除小面积区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(255 - processed_image, connectivity=8)
    for i in range(1, num_labels):  # 跳过背景（标签 0）
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 6000:
            processed_image[labels == i] = 255

    return processed_image

# 批量处理文件夹中的所有图像
def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        if os.path.isfile(input_path) and file_name.lower().endswith('.jpg'):
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"无法读取图像，请检查路径：{input_path}")
            else:
                processed_image = process_image(image)
                output_path = os.path.join(output_folder, file_name)
                cv2.imwrite(output_path, processed_image)
                print(f"处理后的图像已保存到：{output_path}")

if __name__ == "__main__":
    input_folder = "./data"
    output_folder = "./seg_result"
    process_images_in_folder(input_folder, output_folder)