
# Chicken Counting Dataset (v17)

This dataset is designed for object detection tasks involving chicken counting and monitoring. It has been curated and augmented using Roboflow and is formatted in **YOLOv11**.

## 📊 Dataset Split

- **Total Images**: 12,703
- **Train Set**: 11,100 images (87%)
- **Validation Set**: 1,103 images (9%)
- **Test Set**: 500 images (4%)

## 📁 Directory Structure

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
🐔 Classes
nc: 1

names: ['chicken']

🧠 Format
Annotation format: YOLOv11

Each image is labeled for the presence of chickens.

🛠 Pre-processing
Each image in the dataset has undergone the following processing:

Auto-orientation of pixel data (EXIF data stripped)

Resized to 640x640 using "fit" mode (white padding to maintain aspect ratio)

🔄 Augmentation Techniques
Three augmented versions of each source image were generated using:

✅ 50% probability of horizontal flip

🔄 Random rotation between -8° to +8°

💡 Brightness variation between -10% to +10%

🔬 Gaussian blur (0 to 1.5 px)

⚪ Salt-and-pepper noise added to 1.5% of the pixels

🔗 Dataset Source :- https://universe.roboflow.com/thesis-3c51t/chicken-counting/dataset/17

Dataset Download Link
%%bash
cd /content
mkdir -p dataset2
cd dataset2
curl -L "https://universe.roboflow.com/ds/l0JdPqNrPx?key=mt2ji07rnR" -o roboflow.zip
unzip -q roboflow.zip
rm roboflow.zip
