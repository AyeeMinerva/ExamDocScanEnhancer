import cv2
import numpy as np
import os

def remove_shadows(image, output_dir, filename):
    """去除阴影，并保存中间结果"""
    dilated_img = cv2.dilate(image, np.ones((7, 7), np.uint8))
    blurred_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(image, blurred_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_shadow_removed.png"), norm_img)
    return norm_img

def enhance_document(image_path, output_path, intermediate_dir):
    """处理单张图片，并保存中间结果"""
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(intermediate_dir, f"{filename}_original.png"), img)

    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(intermediate_dir, f"{filename}_gray.png"), gray)

    # 阴影去除
    no_shadows = remove_shadows(gray, intermediate_dir, filename)

    # CLAHE 局部对比度增强，clipLimit==-
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(no_shadows)
    cv2.imwrite(os.path.join(intermediate_dir, f"{filename}_clahe_enhanced.png"), enhanced_gray)

    #拉高亮度值
    enhanced_light = cv2.addWeighted(enhanced_gray, 1.1, np.zeros(enhanced_gray.shape, enhanced_gray.dtype), 0, 0)
    cv2.imwrite(os.path.join(intermediate_dir, f"{filename}_light_enhanced.png"), enhanced_light)
    
    # 直接二值化 - 去除背景文字
    _, binary_img = cv2.threshold(enhanced_light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(intermediate_dir, f"{filename}_binary_otsu.png"), binary_img)

    # 保存最终结果
    cv2.imwrite(output_path, binary_img)

def batch_process(input_folder, output_folder, intermediate_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(intermediate_folder):
        os.makedirs(intermediate_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"scanned_{filename}")
            enhance_document(input_path, output_path, intermediate_folder)
            print(f"Processed: {filename}")

# 使用示例
input_folder = "input_images"  # 源文件夹
output_folder = "output_images"  # 输出文件夹
intermediate_folder = "intermediate_images"  # 中间结果文件夹
batch_process(input_folder, output_folder, intermediate_folder)