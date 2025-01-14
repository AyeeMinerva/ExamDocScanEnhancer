import sys
import cv2
import numpy as np
import os
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton


class ImageProcessor(QMainWindow):
    def __init__(self, input_folder, output_folder, intermediate_folder):
        super().__init__()
        
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.intermediate_folder = intermediate_folder
        self.image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.current_index = 0
        self.selected_points = []
        self.dragging_point = -1  # -1表示没有拖动任何点
        
        # 设置UI
        self.setWindowTitle("Document Scanner")
        self.setGeometry(100, 100, 800, 600)
        
        self.image_label = QLabel(self)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        
        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(self.next_image)
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.next_button)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.load_image()

    def load_image(self):
        if self.current_index < len(self.image_files):
            self.image_path = os.path.join(self.input_folder, self.image_files[self.current_index])
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.original_image)
            self.selected_points = []  # Reset selected points
        else:
            print("All images processed")
            self.close()

    def display_image(self, img):
        """显示图片，按比例缩放到 QLabel 大小"""
        self.scaled_image = cv2.resize(img, (self.image_label.width(), self.image_label.height()), interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(self.scaled_image, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_image.shape
        bytes_per_line = 3 * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)

    def mouse_press_event(self, event):
        """鼠标按下，添加或拖动点"""
        x, y = event.x(), event.y()
        img_x, img_y = self.map_to_image_coords(x, y)
        
        if event.button() == Qt.LeftButton:
            # 检查是否点击了现有点
            for i, point in enumerate(self.selected_points):
                px, py = self.map_to_display_coords(point[0], point[1])
                if abs(px - x) < 10 and abs(py - y) < 10:  # 点击点的误差范围
                    self.dragging_point = i
                    return

            # 添加新点
            if len(self.selected_points) < 4:
                self.selected_points.append((img_x, img_y))
                self.update_display()

    def mouse_release_event(self, event):
        """鼠标释放，结束拖动"""
        self.dragging_point = -1

    def mouse_move_event(self, event):
        """鼠标拖动，修改点的位置"""
        if self.dragging_point != -1:
            x, y = event.x(), event.y()
            img_x, img_y = self.map_to_image_coords(x, y)
            self.selected_points[self.dragging_point] = (img_x, img_y)
            self.update_display()

    def update_display(self):
        """更新显示，包括点和框"""
        img_copy = self.original_image.copy()  # 每次都使用原始未标记的图片副本
        img_copy = cv2.resize(img_copy, (self.image_label.width(), self.image_label.height()), interpolation=cv2.INTER_AREA)
        
        for point in self.selected_points:
            px, py = self.map_to_display_coords(point[0], point[1])
            # 更新点的颜色和大小：绿色，半透明，5的大小
            cv2.circle(img_copy, (int(px), int(py)), 5, (34, 177, 76), -1) 
        
        if len(self.selected_points) == 4:
            pts = np.array([self.map_to_display_coords(p[0], p[1]) for p in self.selected_points], dtype=np.int32)
            # 更新框的颜色和粗细：柔和的红色，2的粗细
            cv2.polylines(img_copy, [pts], isClosed=True, color=(255, 255, 0), thickness=2)  
        
        self.display_image(img_copy)

    def map_to_image_coords(self, x, y):
        """将 QLabel 坐标映射到原始图片坐标"""
        img_h, img_w, _ = self.original_image.shape
        label_w, label_h = self.image_label.width(), self.image_label.height()
        scale_x, scale_y = img_w / label_w, img_h / label_h
        return int(x * scale_x), int(y * scale_y)

    def map_to_display_coords(self, x, y):
        """将原始图片坐标映射到 QLabel 坐标"""
        img_h, img_w, _ = self.original_image.shape
        label_w, label_h = self.image_label.width(), self.image_label.height()
        scale_x, scale_y = label_w / img_w, label_h / img_h
        return int(x * scale_x), int(y * scale_y)

    def perform_perspective_transform(self):
        if len(self.selected_points) == 4:
            pts1 = np.float32(self.selected_points)
            # 假设目标矩形的四个角
            width = max(np.linalg.norm(pts1[0] - pts1[1]), np.linalg.norm(pts1[2] - pts1[3]))
            height = max(np.linalg.norm(pts1[0] - pts1[3]), np.linalg.norm(pts1[1] - pts1[2]))
            pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            
            # 透视变换
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            transformed_img = cv2.warpPerspective(self.original_image, matrix, (int(width), int(height)))
            return transformed_img

    def next_image(self):
        if len(self.selected_points) == 4:
            transformed_img = self.perform_perspective_transform()
            self.process_scanned_image(transformed_img)
        self.current_index += 1
        self.load_image()

    def process_scanned_image(self, img):
        # 阴影去除
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        no_shadows = self.remove_shadows(gray)

        # CLAHE 局部对比度增强，clipLimit=0.5
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(no_shadows)

        # 亮度增强
        enhanced_light = cv2.addWeighted(enhanced_gray, 1.1, np.zeros(enhanced_gray.shape, enhanced_gray.dtype), 0, 0)
        
        # 二值化
        _, binary_img = cv2.threshold(enhanced_light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 保存结果
        output_path = os.path.join(self.output_folder, f"scanned_{self.image_files[self.current_index]}")
        cv2.imwrite(output_path, binary_img)

    def remove_shadows(self, image):
        dilated_img = cv2.dilate(image, np.ones((7, 7), np.uint8))
        blurred_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(image, blurred_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return norm_img


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    input_folder = "input_images"  # 源文件夹
    output_folder = "output_images"  # 输出文件夹
    intermediate_folder = "intermediate_images"  # 中间结果文件夹

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(intermediate_folder):
        os.makedirs(intermediate_folder)
    
    window = ImageProcessor(input_folder, output_folder, intermediate_folder)
    window.show()
    
    sys.exit(app.exec_())
