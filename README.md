# ExamDocScanEnhancer

## 项目简介 / Project Introduction

ExamDocScanEnhancer 是一个基于 PyQt5 和 OpenCV 的文档扫描增强工具。该项目允许用户通过图形界面选择图像上的四个点进行透视变换，并对扫描的文档进行去阴影、增强对比度、亮度调整和二值化处理，最终生成清晰的文档图像。

ExamDocScanEnhancer is a document scan enhancement tool based on PyQt5 and OpenCV. This project allows users to select four points on the image via a graphical interface for perspective transformation, and then processes the scanned document by removing shadows, enhancing contrast, adjusting brightness, and performing binarization to produce a clear document image.

## 功能特点 / Features

- **图形界面操作**：用户可以通过图形界面选择图片并标记四个点进行透视变换。
- **阴影去除**：去除图像中的阴影，以提高文档的清晰度。
- **局部对比度增强 (CLAHE)**：增强图像的对比度，突出文本内容。
- **亮度增强**：对图像进行亮度调整，确保文本清晰可见。
- **二值化**：使用 Otsu 方法将图像转为黑白，以便更好地识别和保存文档内容。
- **批量处理**：支持批量处理指定文件夹中的所有图像，自动进行处理并保存结果。

- **Graphical Interface**: Users can select images and mark four points for perspective transformation via the graphical interface.
- **Shadow Removal**: Removes shadows from the image to enhance document clarity.
- **Local Contrast Enhancement (CLAHE)**: Enhances the contrast of the image to highlight text content.
- **Brightness Enhancement**: Adjusts the brightness of the image to ensure clear visibility of text.
- **Binarization**: Converts the image to black and white using the Otsu method for better document preservation.
- **Batch Processing**: Supports batch processing of all images in a specified folder, automatically processing and saving the results.

## 安装 / Installation

1. 克隆该项目：

   Clone this repository:
   ```
   git clone https://github.com/yourusername/ExamDocScanEnhancer.git
   ```

2. 安装依赖库：

   Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. 下载和配置 OpenCV：

   Download and configure OpenCV if not already installed.

## 使用方法 / Usage

### 1. 启动图形界面应用 / Running the GUI Application

- 运行 `mainapp.py` 文件，启动图形界面应用。用户可以加载图像、选择四个点进行透视变换，并处理文档。
  
  Run the `mainapp.py` file to launch the graphical interface. Users can load images, select four points for perspective transformation, and process the document.

```
python mainapp.py
```

### 2. 批量处理 / Batch Processing

- 运行 `picScan.py` 文件，批量处理指定文件夹中的所有图像。

  Run the `picScan.py` file to batch process all images in a specified folder.

```
python picScan.py
```

### 输入文件夹和输出文件夹 / Input and Output Folders

- **输入文件夹 (input_images)**：存放待处理的图像文件。
- **输出文件夹 (output_images)**：存放处理后保存的图像。
- **中间文件夹 (intermediate_images)**：保存每个处理步骤的中间结果，便于调试。

- **Input Folder (input_images)**: The folder containing the images to be processed.
- **Output Folder (output_images)**: The folder to store the processed images.
- **Intermediate Folder (intermediate_images)**: The folder storing intermediate results of each processing step for debugging.

## 代码结构 / Code Structure

- **mainapp.py**: 主界面文件，包含图形界面和透视变换功能。
- **picScan.py**: 提供批量处理和图像增强功能。
- **remove_shadows**: 去除图像阴影的辅助函数。
- **enhance_document**: 处理单张图像的函数，执行图像增强、去阴影、亮度调整等步骤。

- **mainapp.py**: Main interface file containing the graphical interface and perspective transformation functionality.
- **picScan.py**: Provides batch processing and image enhancement features.
- **remove_shadows**: Helper function for removing shadows from the image.
- **enhance_document**: Function to process a single image, performing enhancement, shadow removal, brightness adjustment, etc.

## 许可证 / License

该项目使用 AGPL V3.0 许可证，详细信息请参阅 LICENSE 文件。

This project is licensed under the AGPL V3.0 License. See the LICENSE file for details.
