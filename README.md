# Fruit Sorting System by Color and Size

An automated fruit sorting system based on color and size classification using Digital Image Processing techniques. Developed with Python, OpenCV, and Streamlit.

> **Accuracy Target:** >95% (Achieved **96.80%** on benchmark dataset)

---

## 🚀 Technical Pipeline (5 Steps)

1. **Pre-processing**: Histogram Equalization & Noise Reduction (Gaussian/Median Filter)
2. **Color Segmentation**: BGR to HSV conversion & Masking combined with Otsu's Binarization
3. **Morphological & Contours**: Opening and Closing operations to clean masks and find object boundaries
4. **Feature Extraction**: Extracting object features including Area, Circularity, and Dominant Hue (HSV)
5. **Classification**: Rule-based grading for color and size

## 📁 Project Structure

- `app.py`: Interactive Streamlit Web UI application.
- `pipeline.py`: Core image processing pipeline.
- `evaluate.py`: Batch evaluation script to benchmark classification accuracy.
- `requirements.txt`: List of Python dependencies.
- `data/`: Local directory containing fruit images dataset (ignored).

## 🛠️ How to Run

### 1. Setup Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit UI Application
```bash
streamlit run app.py
```

### 4. Run Accuracy Evaluation
```bash
python evaluate.py
```

---
© 2026 Fruit Sorting Project
