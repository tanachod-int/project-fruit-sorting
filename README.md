# Fruit Sorting System by Color and Size
## Project for CS483 Digital Image Processing

ระบบคัดแยกผลไม้จากสีและขนาดโดยใช้เทคนิค Image Processing สำหรับงานอุตสาหกรรม พัฒนาด้วย Python และ OpenCV บน Google Colab

> **Accuracy Target:** >95%

---

## 🚀 Technical Pipeline (5 Steps)

1.  **Pre-processing**: Histogram Equalization & Noise Reduction (Gaussian/Median Filter)
2.  **Color Segmentation**: BGR to HSV conversion & Masking using Otsu's Binarization
3.  **Morphological & Contours**: Erosion, Dilation, Opening, Closing to clean masks and find object boundaries
4.  **Feature Extraction**: Measuring Area, Perimeter, Circularity, and Average Color (HSV)
5.  **Classification**: Rule-based analysis for color, size, and quality sorting

## 📁 Project Structure

- `data/`: Sample images and datasets (not uploaded to Git)
- `notebooks/`: Core project notebook (.ipynb)
- `results/`: Annotated results and accuracy reports
- `plans/`: Implementation plans and technical notes
- `doc/`: Supporting documents and reference materials

## 🛠️ Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

---
© 2026 CS483 Senior Project
