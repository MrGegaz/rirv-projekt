# Traffic Signs Analysis <br> (Analiza prometnih znakova)

## General
- Traffic sign detection is a computer vision technology used in Advanced Driver Assistance Systems (ADAS) and autonomous vehicles to automatically detect, classify, and interpret traffic signs from video feeds. It employs deep learning methods like YOLO, SSD, and CNNs to identify speed limits, warnings, and stop signs in real-time to enhance safety.

- Deep Learning Models: State-of-the-art systems heavily utilize Convolutional Neural Networks (CNN) for high accuracy, specifically models like YOLOv5, YOLOv7, and YOLOv8 for rapid, real-time detection.

### Detection Techniques
- **Color-based**: Using color space conversion (RGB to HSV or HSI) to segment signs by their distinct colors, often red, yellow, or blue.
- **Shape-based**: Detecting edges to identify geometric shapes like triangles, circles, and squares, which often indicates specific sign types.
- **Machine Learning/AI**: Training models to identify sign features,, including techniques using Support Vector Machines (SVM) and CNNs for classification.

### Components of a Traffic Sign Detection System
1. **Image Acquisition**: A vehicle-mounted camera captures road images
2. **Detection/Segmentation**: The algorithm identifies potential traffic signs in the image while ignoring background noise.
3. **Classification**: The detected object is classified into specific categories (e.g., speed limit 50, stop).
4. **Display/Action**: The system informs the driver via the dashboard or adjusts vehicle speed automatically.

### Challenges and Solutions
- **Environmental Obstacles**: Poor weather, poor lighting, or damaged signs can degrade performance
- **Small Target Recognition**: Detecting small signs at a distance is challenging, which is addressed by adopting improved feature extraction methods like MobileNetV2.
- **Computational Efficiency**: Ensuring algorithms are lightweight enough to operate efficiently on limited hardware is a key research area, using methods like the YOLO-SAL model

## [Dataset](https://www.kaggle.com/datasets/pkdarabi/cardetection/data)

- **Name of Classes**: Green Light, Red Light, Speed Limit 10, Speed Limit 100, Speed Limit 110, Speed Limit 120, Speed Limit 20, Speed Limit 30, Speed Limit 40, Speed Limit 50, Speed Limit 60, Speed Limit 70, Speed Limit 80, Speed Limit 90, Stop

## Implementation
- [YOLO-BS](https://www.nature.com/articles/s41598-025-88184-0): a traffic sign detection algorithm based on YOLOv8


## Korisni linkovi
- https://www.kaggle.com/datasets/pkdarabi/cardetection/data
- https://www.nature.com/articles/s41598-025-88184-0