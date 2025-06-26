# Hen Health Detection Dataset (v2)

This dataset is created for training object detection models to distinguish between healthy and sick hens. The dataset has been structured and augmented using Roboflow and follows the **YOLOv11** annotation format.

## 📊 Dataset Split

- **Total Images**: 7,080  
- **Train Set**: 6,292 images (89%)  
- **Validation Set**: 592 images (8%)  
- **Test Set**: 196 images (3%)

## 📁 Directory Structure

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
🐓 Classes
nc: 2

names: ['Healthy', 'Sick']

🧠 Format
Annotation format: YOLOv11

Each image is labeled with either a "Healthy" or "Sick" chicken.

🛠 Pre-processing
Images were standardized using the following transformations:

Auto-orientation of pixel data (EXIF metadata removed)

Resized to 640x640 using stretch mode

🔄 Augmentation Techniques
Two augmented versions were created per original image using:

🔄 Random rotation between -10° to +10°

📦 Bounding Box Transformations
⚪ Salt-and-pepper noise applied to 4% of pixels

🔗 Dataset Information
Workspace: v1-tgwsx

Project: hen-v2-dh1mf

Version: 1

License: CC BY 4.0

Dataset URL: https://universe.roboflow.com/v1-tgwsx/hen-v2-dh1mf/dataset/1#

⬇️ Dataset Download Instructions (Colab)

%%bash
cd /content
mkdir -p dataset3
cd dataset3
curl -L "https://universe.roboflow.com/ds/2iDgDUKL5w?key=Qt1UfrmPN7" -o roboflow.zip
unzip -q roboflow.zip
rm roboflow.zip
