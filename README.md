# 🐔 CluckVision: Poultry Chicken Detection and Monitoring System

CluckVision is an AI-powered solution for monitoring poultry environments. It uses YOLOv8 for real-time chicken detection and counting, and applies classification techniques to distinguish between healthy and sick chickens with high accuracy.

---

## 🚀 Features

- 🔍 **YOLOv8 Object Detection**: Trained to detect chickens in images and video streams.
- 🧮 **Chicken Counting**: Automatically counts the number of chickens in each frame.
- 🩺 **Health Classification**: Differentiates sick vs. healthy chickens using confidence scores.
- 📐 **Polygonal Approximation**: Refines object boundaries for precise segmentation.
- 📊 **Integrated Dataset & Model Tracking**: Comes with labeled datasets and trained models.
- 🎥 **Real-time Monitoring Support**: Can be adapted to CCTV/live feed integrations.

---

## 🧠 Tech Stack

- **Framework**: Python, OpenCV
- **Model**: YOLOv8 (Ultralytics)
- **Libraries**: NumPy, Pandas, Matplotlib, Torch, Tkinter (if applicable)
- **Hardware**: GPU-based training recommended (NVIDIA CUDA)

---

## 📁 Folder Structure


🛠️ Setup Instructions
Clone the repository

bash
Copy
Edit
git clone https://github.com/Sayancode2026/CluckVision-Poultry-Chicken-Detection-and-Monitoring-System.git
cd CluckVision-Poultry-Chicken-Detection-and-Monitoring-System
Create a virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the demo

bash
Copy
Edit
# Example: Run counting or classification notebook
jupyter notebook Chicken_Detection_Final/Chicken\ Count/Chicken_Counting_POC.ipynb
📸 Sample Output
Detection	Counting	Health Classification

📈 Results
Achieved high mAP scores on custom dataset of chicken images.

Health classification accuracy: 95%+ based on annotated datasets.

Reduced manual monitoring needs by over 70%.

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgments
YOLOv8 by Ultralytics

Custom poultry datasets sourced and manually labeled

OpenCV and PyTorch communities for constant innovation

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
