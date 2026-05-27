"""
Fruit Sorting Pipeline — Core Functions
========================================
ฟังก์ชันหลักของระบบคัดแยกผลไม้ ใช้ร่วมกันระหว่าง Notebook (Colab) และ Streamlit UI

Pipeline 5 ขั้นตอน:
  1. Pre-processing (Histogram EQ + Noise Reduction)
  2. Color Segmentation (BGR→HSV + Masking + Otsu)
  3. Morphological Operations & Contour Detection
  4. Feature Extraction (Area, Color)
  5. Classification (Rule-based)
"""

import cv2
import numpy as np
from typing import Tuple

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
GAUSSIAN_KERNEL = (5, 5)
MEDIAN_KSIZE = 5

# Size thresholds (now dynamically calculated)

# HSV color ranges for masking
COLOR_RANGES = {
    'red':    {'lower1': (0, 100, 100),   'upper1': (15, 255, 255),
               'lower2': (160, 100, 100), 'upper2': (180, 255, 255)},
    'orange': {'lower1': (16, 100, 100),  'upper1': (22, 255, 255)},
    'yellow': {'lower1': (23, 100, 100),  'upper1': (38, 255, 255)},
    'green':  {'lower1': (39, 50, 50),    'upper1': (90, 255, 255)},
    'purple': {'lower1': (91, 50, 50),    'upper1': (159, 255, 255)},
}

# Color classification label by hue range
COLOR_LABELS = [
    ((0, 15),    "Red (Strawberry/Apple)"),
    ((160, 180), "Red (Strawberry/Apple)"),
    ((16, 22),   "Orange (Orange/Mango)"),
    ((23, 38),   "Yellow (Banana/Mango)"),
    ((39, 90),   "Green (Green Grape/Mango)"),
    ((91, 159),  "Purple (Grape)"),
]


# ──────────────────────────────────────────────
# Step 1: Pre-processing
# ──────────────────────────────────────────────
def manual_histogram_equalization(gray_img: np.ndarray) -> np.ndarray:
    """
    Global Histogram Equalization (manual implementation)
    
    สูตร: s = (CDF(r) - CDF_min) / (N - CDF_min) * (L-1)
    โดย:  r = pixel intensity ต้นทาง
          N = จำนวน pixel ทั้งหมด
          L = 256 (จำนวนระดับสี)
    
    ผลลัพธ์: กระจาย intensity ให้สม่ำเสมอขึ้น ทำให้ contrast ดีขึ้น
    """
    if len(gray_img.shape) != 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    hist, _ = np.histogram(gray_img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    equalized_lookup_table = np.ma.filled(cdf_m, 0).astype('uint8')
    return equalized_lookup_table[gray_img]


def denoise_image(img: np.ndarray, method: str = 'gaussian') -> np.ndarray:
    """
    Reduces noise using specified method.
    'gaussian': Smooths general electronic noise.
    'median': Best for salt-and-pepper noise.
    """
    if method.lower() == 'gaussian':
        return cv2.GaussianBlur(img, GAUSSIAN_KERNEL, 0)
    elif method.lower() == 'median':
        return cv2.medianBlur(img, MEDIAN_KSIZE)
    else:
        return img


def compute_histogram_cdf(gray_img: np.ndarray) -> dict:
    """Compute histogram (PDF) and CDF for visualization."""
    if len(gray_img.shape) != 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(gray_img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    return {'hist': hist, 'cdf': cdf_normalized, 'bins': bins[:-1]}


# ──────────────────────────────────────────────
# Step 2: Color Segmentation
# ──────────────────────────────────────────────
def create_color_mask(bgr_img: np.ndarray, color_type: str = 'red') -> np.ndarray:
    """Converts BGR to HSV and returns a binary mask for the specified color."""
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    color = color_type.lower()

    if color not in COLOR_RANGES:
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    ranges = COLOR_RANGES[color]
    mask = cv2.inRange(hsv_img, np.array(ranges['lower1']), np.array(ranges['upper1']))

    # Red color wraps around the hue circle
    if 'lower2' in ranges:
        mask2 = cv2.inRange(hsv_img, np.array(ranges['lower2']), np.array(ranges['upper2']))
        mask = cv2.bitwise_or(mask, mask2)

    return mask




def manual_otsu_thresholding(gray_img: np.ndarray) -> np.ndarray:
    """
    Manual implementation of Otsu's Thresholding algorithm
    to demonstrate Technical Depth (Math/Logic).
    """
    hist, _ = np.histogram(gray_img.flatten(), 256, [0, 256])
    total_pixels = gray_img.size
    
    current_max_var = 0
    best_threshold = 0
    
    sum_total = np.dot(np.arange(256), hist)
    
    weight_bg = 0
    sum_bg = 0
    
    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
            
        weight_fg = total_pixels - weight_bg
        if weight_fg == 0:
            break
            
        sum_bg += i * hist[i]
        
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        
        # Between-class variance
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        
        if var_between > current_max_var:
            current_max_var = var_between
            best_threshold = i
            
    binary_mask = np.where(gray_img > best_threshold, 255, 0).astype(np.uint8)
    return binary_mask


def otsu_segmentation(img: np.ndarray) -> np.ndarray:
    """Uses custom Otsu's thresholding to segment object from background."""
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    blurred = cv2.GaussianBlur(gray_img, GAUSSIAN_KERNEL, 0)
    
    binary_mask = manual_otsu_thresholding(blurred)
    
    # Check background color by looking at the borders
    top_border = binary_mask[0, :]
    bottom_border = binary_mask[-1, :]
    left_border = binary_mask[:, 0]
    right_border = binary_mask[:, -1]
    
    # Calculate the average pixel value of the borders
    border_mean = np.mean(np.concatenate([top_border, bottom_border, left_border, right_border]))
    
    # If the border mean is closer to white (>127), it means the background is white.
    # In OpenCV contours, background should be black (0), so we must invert it.
    if border_mean > 127:
        return cv2.bitwise_not(binary_mask)
    
    return binary_mask


# ──────────────────────────────────────────────
# Step 3: Morphological Operations & Contours
# ──────────────────────────────────────────────
def apply_morphology(binary_mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Cleans up the binary mask using Opening (remove noise) and Closing (fill holes)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    return cleaned_mask


def find_fruit_contours(cleaned_mask: np.ndarray, original_img: np.ndarray,
                        min_area: int = 500, min_circularity: float = 0.0) -> Tuple[np.ndarray, list]:
    """Finds contours from the mask and filters out small noise and non-fruit shapes."""
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= min_area:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity >= min_circularity:
            valid_contours.append(cnt)
            
    output_img = original_img.copy()
    cv2.drawContours(output_img, valid_contours, -1, (0, 255, 0), 2)
    return output_img, valid_contours


# ──────────────────────────────────────────────
# Step 4: Feature Extraction
# ──────────────────────────────────────────────
def extract_features(contour: np.ndarray, bgr_img: np.ndarray) -> dict:
    """Extracts mathematical features from a contour such as area and mean color."""
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)

    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Calculate Mean Saturation and Value
    mean_color = cv2.mean(hsv_img, mask=mask)
    
    # Hue calculation using Histogram to find Dominant Hue
    # Filters out low Saturation and Value to reduce noise
    s_channel = hsv_img[:, :, 1]
    v_channel = hsv_img[:, :, 2]
    color_mask = cv2.bitwise_and(mask, cv2.bitwise_and((s_channel > 50).astype(np.uint8)*255, (v_channel > 50).astype(np.uint8)*255))
    
    hist = cv2.calcHist([hsv_img], [0], color_mask, [180], [0, 180])
    # If the fruit is entirely black/white (color_mask is empty), fallback to original mask
    if hist.sum() == 0:
        hist = cv2.calcHist([hsv_img], [0], mask, [180], [0, 180])
        
    dominant_hue = int(np.argmax(hist))

    return {
        'area': area,
        'mean_hue': dominant_hue,  # Now represents dominant hue
        'mean_saturation': mean_color[1],
        'mean_value': mean_color[2],
        'bounding_box': (x, y, w, h),
    }


# ──────────────────────────────────────────────
# Step 5: Classification
# ──────────────────────────────────────────────
def classify_color(hue: float) -> str:
    """Classify fruit color based on HSV Hue value."""
    for (low, high), label in COLOR_LABELS:
        if low <= hue <= high:
            return label
    return "Unknown"


def classify_size(area: float,
                  small_thresh: float,
                  medium_thresh: float) -> str:
    """Classify fruit size based on contour area."""
    if area < small_thresh:
        return "Small"
    elif area < medium_thresh:
        return "Medium"
    else:
        return "Large"


def classify_fruit(features: dict,
                   small_thresh: float,
                   medium_thresh: float) -> dict:
    """Full classification: color and size."""
    return {
        'color': classify_color(features['mean_hue']),
        'size': classify_size(features['area'], small_thresh, medium_thresh),
    }


# ──────────────────────────────────────────────
# Full Pipeline (end-to-end)
# ──────────────────────────────────────────────
def run_pipeline(bgr_img: np.ndarray, min_area: int = 500,
                 min_circularity: float = 0.0,
                 morph_kernel: int = 5,
                 denoise_method: str = 'gaussian',
                 small_thresh: float = None,
                 medium_thresh: float = None) -> dict:
    """
    Runs the complete fruit-sorting pipeline on a single BGR image.

    Returns a dict with intermediate results for each step + final classifications.
    """
    # Calculate dynamic size thresholds based on image area if not provided
    img_area = bgr_img.shape[0] * bgr_img.shape[1]
    if small_thresh is None:
        small_thresh = img_area * 0.02  # 2% of image
    if medium_thresh is None:
        medium_thresh = img_area * 0.08 # 8% of image

    # Step 1: Pre-processing
    denoised = denoise_image(bgr_img, method=denoise_method)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    equalized = manual_histogram_equalization(gray)

    # Histogram data for visualization
    hist_before = compute_histogram_cdf(gray)
    hist_after = compute_histogram_cdf(equalized)

    # Step 2: Segmentation (Otsu + Color Hybrid)
    otsu_mask = otsu_segmentation(denoised)

    # Color masks for visualization + segmentation
    color_masks = {}
    combined_color_mask = np.zeros(denoised.shape[:2], dtype=np.uint8)
    for color_name in COLOR_RANGES:
        cmask = create_color_mask(denoised, color_name)
        color_masks[color_name] = cmask
        combined_color_mask = cv2.bitwise_or(combined_color_mask, cmask)

    # รวม Otsu + Color Mask เพื่อความแม่นยำ
    segmentation_mask = cv2.bitwise_or(otsu_mask, combined_color_mask)

    # Step 3: Morphology & Contours
    cleaned_mask = apply_morphology(segmentation_mask, kernel_size=morph_kernel)
    contour_img, contours = find_fruit_contours(cleaned_mask, bgr_img, min_area=min_area, min_circularity=min_circularity)

    # Step 4 & 5: Feature Extraction + Classification
    results = []
    annotated_img = bgr_img.copy()

    # สกัด Feature ทั้งหมด
    all_features = [extract_features(cnt, bgr_img) for cnt in contours]

    for i, (cnt, features) in enumerate(zip(contours, all_features)):
        classification = classify_fruit(features, small_thresh, medium_thresh)

        x, y, w, h = features['bounding_box']

        # Choose color for bounding box based on classification
        color_map = {
            'Red': (0, 0, 255),
            'Orange': (0, 128, 255),
            'Yellow': (0, 255, 255),
            'Green': (0, 255, 0),
            'Purple': (180, 0, 255),
        }
        box_color = (0, 255, 0)
        for key, rgb in color_map.items():
            if key in classification['color']:
                box_color = rgb
                break

        # Draw bounding box and label
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, 2)

        label = f"#{i+1} {classification['color'].split('(')[0].strip()} | {classification['size']}"

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_img, (x, y - th - 10), (x + tw + 4, y), box_color, -1)
        cv2.putText(annotated_img, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        results.append({
            'id': i + 1,
            'features': features,
            'classification': classification,
        })

    return {
        'step1_denoised': denoised,
        'step1_equalized': equalized,
        'step1_hist_before': hist_before,
        'step1_hist_after': hist_after,
        'step2_otsu_mask': otsu_mask,
        'step2_color_masks': color_masks,
        'step3_cleaned_mask': cleaned_mask,
        'step3_contour_img': contour_img,
        'annotated_img': annotated_img,
        'contours': contours,
        'results': results,
    }
