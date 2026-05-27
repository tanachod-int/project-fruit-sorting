"""
🍎 Fruit Sorting System — Streamlit UI
========================================
ระบบคัดแยกผลไม้จากสีและขนาดโดยใช้เทคนิค Image Processing

วิธีรัน:
  streamlit run app.py

"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pipeline import run_pipeline

# ─────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Eyesort",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
        font-weight: 300;
    }

    /* Pipeline step cards */
    .step-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .step-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    .step-card h4 {
        color: #a78bfa;
        margin: 0 0 0.3rem 0;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .step-card p {
        color: #94a3b8;
        margin: 0;
        font-size: 0.8rem;
        line-height: 1.4;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #252540);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .metric-card .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }

    /* Divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #a78bfa, transparent);
        border: none;
        margin: 1.5rem 0;
        border-radius: 2px;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# Helper: Convert between PIL <-> OpenCV
# ─────────────────────────────────────────────────────
def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format."""
    rgb = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_display(cv2_img: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB for Streamlit display."""
    if len(cv2_img.shape) == 2:
        return cv2_img  # Grayscale
    return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🍎 Project Eyesort</h1>
    <p>ระบบวิเคราะห์และคัดเกรดผลไม้อัตโนมัติด้วยคุณลักษณะสีและขนาด/p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# Sidebar — Parameters & Upload
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Parameters")
    st.caption("ปรับค่าต่างๆ เพื่อดูผลลัพธ์แบบ Real-time")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Denoise method
    denoise_method = st.selectbox(
        "🔇 Noise Reduction Method",
        options=['gaussian', 'median'],
        help="Gaussian: ลด noise ทั่วไป | Median: ลด salt-and-pepper noise"
    )

    # Morphology kernel
    morph_kernel = st.slider(
        "🔘 Morphology Kernel Size",
        min_value=3, max_value=15, value=5, step=2,
        help="ขนาด kernel สำหรับ Opening/Closing — ค่ามากขึ้น = ทำความสะอาด mask มากขึ้น"
    )

    # Min area
    min_area = st.slider(
        "📐 Min Contour Area",
        min_value=100, max_value=10000, value=500, step=100,
        help="พื้นที่ขั้นต่ำของ contour (pixels) — กรอง noise เล็กๆ ออก"
    )

    # Min circularity
    min_circularity = st.slider(
        "⭕ Min Circularity (Shape)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="0.0 = ไม่กรองรูปร่าง | 1.0 = วงกลมสมบูรณ์แบบ — ใช้กรองวัตถุที่ไม่ใช่ผลไม้ (เช่น เส้นยาวๆ หรือสี่เหลี่ยม)"
    )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🏷️ Classification Thresholds")

    small_thresh_pct = st.number_input(
        "Small → Medium (% of Image Area)",
        min_value=0.5, max_value=20.0, value=2.0, step=0.5,
    )

    medium_thresh_pct = st.number_input(
        "Medium → Large (% of Image Area)",
        min_value=1.0, max_value=50.0, value=8.0, step=1.0,
    )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📸 Upload Image")

    uploaded_file = st.file_uploader(
        "เลือกรูปภาพผลไม้",
        type=["jpg", "jpeg", "png", "bmp"],
        help="รองรับ JPG, PNG, BMP"
    )

# ─────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────
if uploaded_file is not None:
    # Load image
    pil_img = Image.open(uploaded_file)
    bgr_img = pil_to_cv2(pil_img)

    # Run pipeline
    with st.spinner("🔄 กำลังประมวลผล Pipeline..."):
        img_area = bgr_img.shape[0] * bgr_img.shape[1]
        pipeline_result = run_pipeline(
            bgr_img,
            min_area=min_area,
            min_circularity=min_circularity,
            morph_kernel=morph_kernel,
            denoise_method=denoise_method,
            small_thresh=img_area * (small_thresh_pct / 100.0),
            medium_thresh=img_area * (medium_thresh_pct / 100.0),
        )

    results = pipeline_result['results']

    # ─── Metrics Row ───
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(results)}</p>
            <p class="metric-label">🍊 ผลไม้ที่ตรวจพบ</p>
        </div>
        """, unsafe_allow_html=True)

    color_counts = {}
    size_counts = {'Small': 0, 'Medium': 0, 'Large': 0}

    for r in results:
        c = r['classification']
        color_key = c['color'].split('(')[0].strip()
        color_counts[color_key] = color_counts.get(color_key, 0) + 1
        size_counts[c['size']] += 1

    with m2:
        dominant_color = max(color_counts, key=color_counts.get) if color_counts else "—"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{dominant_color}</p>
            <p class="metric-label">🎨 สีที่พบมากที่สุด</p>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        dominant_size = max(size_counts, key=size_counts.get) if any(size_counts.values()) else "—"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{dominant_size}</p>
            <p class="metric-label">📏 ขนาดที่พบมากที่สุด</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ─── Tab Layout ───
    tab_result, tab_pipeline = st.tabs([
        "🖼️ ผลลัพธ์การคัดแยก",
        "🔬 Pipeline Step-by-Step",
    ])

    # ━━━ TAB 1: Results ━━━
    with tab_result:
        col_orig, col_result = st.columns(2, gap="medium")

        with col_orig:
            st.markdown("#### 📷 ภาพต้นฉบับ")
            st.image(cv2_to_display(bgr_img), use_container_width=True)

        with col_result:
            st.markdown("#### 🏷️ ภาพผลลัพธ์ (Annotated)")
            st.image(cv2_to_display(pipeline_result['annotated_img']), use_container_width=True)

        # Result cards
        if results:
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            st.markdown("#### 📋 รายละเอียดผลไม้แต่ละลูก")

            # Display results in columns (max 4 per row)
            cols_per_row = min(len(results), 4)
            for row_start in range(0, len(results), cols_per_row):
                row_items = results[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)

                for idx, item in enumerate(row_items):
                    with cols[idx]:
                        cls = item['classification']
                        feat = item['features']

                        # Crop the fruit from the image for display
                        x, y, w, h = feat['bounding_box']
                        pad = 10
                        y1 = max(0, y - pad)
                        y2 = min(bgr_img.shape[0], y + h + pad)
                        x1 = max(0, x - pad)
                        x2 = min(bgr_img.shape[1], x + w + pad)
                        cropped = bgr_img[y1:y2, x1:x2]

                        st.image(cv2_to_display(cropped), use_container_width=True,
                                 caption=f"Fruit #{item['id']}")

                        st.markdown(f"""
                        **สี:** {cls['color']}  
                        **ขนาด:** {cls['size']} <span style="font-size: 0.85em; color: #94a3b8;">(พื้นที่ {feat['area']:,.0f} พิกเซล)</span>
                        """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ ไม่พบผลไม้ในภาพ — ลองปรับค่า Min Contour Area ให้ต่ำลง หรือใช้ภาพที่มีพื้นหลังเรียบ")

    # ━━━ TAB 2: Pipeline Step-by-Step ━━━
    with tab_pipeline:
        st.markdown("### 🔬 แสดงผลแต่ละขั้นตอนของ Pipeline")
        st.caption("ดูว่าภาพถูกประมวลผลอย่างไรในแต่ละ Step")

        # Step 1
        st.markdown("""
        <div class="step-card">
            <h4>Step 1 — Pre-processing & Analysis</h4>
            <p>Noise Reduction (ลดจุดรบกวน) + Histogram Analysis (สำหรับวิเคราะห์ Contrast ของภาพ)</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2_to_display(pipeline_result['step1_denoised']),
                     caption=f"After {denoise_method.title()} Denoise", use_container_width=True)
        with c2:
            st.image(pipeline_result['step1_equalized'],
                     caption="Histogram Equalization (Grayscale)", use_container_width=True)

        # Histogram + CDF Chart
        st.markdown("#### 📈 Histogram & CDF Analysis (Visual Contrast Check)")
        st.caption("กราฟนี้ใช้สำหรับวิเคราะห์การกระจายตัวของแสง (Contrast) ในภาพต้นฉบับเทียบกับหลังทำ Equalization เพื่อเป็นข้อมูลประกอบการจูน Threshold (ไม่ได้ส่งภาพ Equalized เข้า Otsu โดยตรงเพื่อป้องกัน Noise Amplify)")
        hist_before = pipeline_result['step1_hist_before']
        hist_after = pipeline_result['step1_hist_after']

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor='#0e1117')
        for ax in axes:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='#94a3b8', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#334155')

        axes[0].bar(hist_before['bins'], hist_before['hist'], color='#667eea', alpha=0.7, width=1)
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(hist_before['bins'], hist_before['cdf'], color='#f97316', linewidth=2)
        ax0_twin.tick_params(colors='#94a3b8', labelsize=8)
        axes[0].set_title('Before Equalization', color='white', fontsize=11)

        axes[1].bar(hist_after['bins'], hist_after['hist'], color='#a78bfa', alpha=0.7, width=1)
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(hist_after['bins'], hist_after['cdf'], color='#f97316', linewidth=2)
        ax1_twin.tick_params(colors='#94a3b8', labelsize=8)
        axes[1].set_title('After Equalization', color='white', fontsize=11)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Step 2
        st.markdown("""
        <div class="step-card">
            <h4>Step 2 — Segmentation (Otsu's + Color Masking)</h4>
            <p>BGR → HSV Conversion + Otsu's Thresholding + Color-based Masking เพื่อแยกวัตถุจากพื้นหลัง</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2_to_display(bgr_img),
                     caption="Original", use_container_width=True)
        with c2:
            st.image(pipeline_result['step2_otsu_mask'],
                     caption="Otsu's Binary Mask", use_container_width=True)

        # Color Masks Grid
        st.markdown("#### 🎨 HSV Color Masks")
        color_masks = pipeline_result['step2_color_masks']
        mask_cols = st.columns(len(color_masks))
        color_emoji = {'red': '🔴', 'orange': '🟠', 'yellow': '🟡', 'green': '🟢', 'purple': '🟣'}
        for i, (cname, cmask) in enumerate(color_masks.items()):
            with mask_cols[i]:
                st.image(cmask, caption=f"{color_emoji.get(cname, '')} {cname.title()} Mask", use_container_width=True)

        # Step 3
        st.markdown("""
        <div class="step-card">
            <h4>Step 3 — Morphological Operations & Contours</h4>
            <p>Opening/Closing เพื่อทำความสะอาด Mask แล้วใช้ findContours() หาขอบเขตวัตถุ</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.image(pipeline_result['step3_cleaned_mask'],
                     caption=f"Cleaned Mask (kernel={morph_kernel})", use_container_width=True)
        with c2:
            st.image(cv2_to_display(pipeline_result['step3_contour_img']),
                     caption=f"Detected Contours ({len(results)} found)", use_container_width=True)

        # Step 4
        st.markdown("""
        <div class="step-card">
            <h4>Step 4 — Feature Extraction</h4>
            <p>สกัด Feature จากแต่ละ Contour: Contour Area (พื้นที่), Dominant Hue (จาก HSV Histogram), Mean Saturation & Value</p>
        </div>
        """, unsafe_allow_html=True)

        if results:
            feat_rows = []
            for item in results:
                f = item['features']
                feat_rows.append({
                    'Fruit #':      item['id'],
                    'Area (px)':    f"{f['area']:,.0f}",
                    'Dominant Hue': f"{f['mean_hue']:.1f}",
                    'Saturation':   f"{f['mean_saturation']:.1f}",
                    'Value':        f"{f['mean_value']:.1f}",
                })
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)
        else:
            st.info("ไม่พบ Contour — ไม่มี Feature ให้แสดง")

        # Step 5
        st.markdown("""
        <div class="step-card">
            <h4>Step 5 — Classification (Rule-based)</h4>
            <p>จำแนก Color จาก Dominant Hue และ Size จาก Area เทียบกับ Threshold ที่กำหนด</p>
        </div>
        """, unsafe_allow_html=True)

        st.image(cv2_to_display(pipeline_result['annotated_img']),
                 caption="Final Annotated Result", use_container_width=True)

        if results:
            cls_rows = []
            for item in results:
                c = item['classification']
                f = item['features']
                cls_rows.append({
                    'Fruit #': item['id'],
                    'Color':   c['color'],
                    'Size':    c['size'],
                    'Area (px)': f"{f['area']:,.0f}",
                })
            st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)

else:
    # ─── Landing / Welcome ───
    st.markdown("---")

    col_welcome, col_info = st.columns([3, 2], gap="large")

    with col_welcome:
        st.markdown("### 👋 ยินดีต้อนรับ!")
        st.markdown("""
        ระบบนี้ใช้เทคนิค **Image Processing** คัดแยกผลไม้จาก:
        - 🎨 **สี** — แยกเป็นกลุ่ม Red, Orange, Yellow, Green, Purple
        - 📏 **ขนาด** — จัดเกรดเป็น Small, Medium, Large

        **เริ่มต้นใช้งาน:** อัปโหลดรูปภาพผลไม้ทาง Sidebar ด้านซ้ายมือ ←
        """)

    with col_info:
        st.markdown("### 🔬 Pipeline 5 ขั้นตอน")

        steps = [
            ("Step 1", "Pre-processing", "Noise Reduction + Hist EQ (Analysis)"),
            ("Step 2", "Color Segmentation", "BGR→HSV + Otsu's Binarization"),
            ("Step 3", "Morphology & Contours", "Opening/Closing + findContours"),
            ("Step 4", "Feature Extraction", "Area, Color"),
            ("Step 5", "Classification", "Rule-based Color/Size"),
        ]

        for num, title, desc in steps:
            st.markdown(f"""
            <div class="step-card">
                <h4>{num} — {title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ───
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:0.8rem;'>"
    "© 2026 Digital Image Processing — Fruit Sorting System"
    "</p>",
    unsafe_allow_html=True,
)
