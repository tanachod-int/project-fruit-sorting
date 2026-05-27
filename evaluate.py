import os
import glob
import cv2
import pandas as pd
from pipeline import run_pipeline

# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────
TEST_DIR = r"data\Fruits Classification"
MAX_IMAGES_PER_CLASS = 50

# Map folder names to acceptable predicted colors
GROUND_TRUTH_MAP = {
    'Apple': ['Red', 'Green'],
    'Strawberry': ['Red'],
    'Banana': ['Yellow', 'Green'],
    'Mango': ['Orange', 'Yellow',  'Green'],
    'Grape': ['Purple', 'Green'],
}

def evaluate_system():
    print(f"=== Starting Evaluation (Color & Size Sorting) ===")
    print(f"Testing directory: {TEST_DIR}\n")
    
    if not os.path.exists(TEST_DIR):
        print(f"Error: Directory '{TEST_DIR}' not found.")
        return

    total_images_tested = 0
    total_color_correct = 0
    results_list = []

    # Loop through each fruit folder
    for fruit_name, valid_colors in GROUND_TRUTH_MAP.items():
        folder_path = os.path.join(TEST_DIR, fruit_name)
        if not os.path.exists(folder_path):
            print(f"Skipping {fruit_name}: Folder not found.")
            continue

        image_files = glob.glob(os.path.join(folder_path, "*.*"))
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:MAX_IMAGES_PER_CLASS]
        
        class_total = 0
        class_color_correct = 0

        for img_path in image_files:
            bgr_img = cv2.imread(img_path)
            if bgr_img is None: continue

            try:
                pipeline_result = run_pipeline(bgr_img)
                detected_fruits = pipeline_result['results']
                
                if not detected_fruits:
                    class_total += 1
                    continue # Failed to detect
                
                main_fruit = detected_fruits[0]['classification']
                predicted_color = main_fruit['color'].split('(')[0].strip()
                
                # 1. Evaluate Color
                if predicted_color in valid_colors:
                    class_color_correct += 1
            except Exception as e:
                print(f"Error on {img_path}: {e}")
                
            class_total += 1

        if class_total > 0:
            color_acc = (class_color_correct / class_total) * 100
            
            print(f"{fruit_name:<15} | Tested: {class_total:<3} | Accuracy: {color_acc:>6.2f}%")
            
            results_list.append({
                'Fruit Class': fruit_name,
                'Tested': class_total,
                'Color Correct': class_color_correct,
                'Accuracy (%)': round(color_acc, 2)
            })
            
            total_images_tested += class_total
            total_color_correct += class_color_correct

    print("-" * 50)
    if total_images_tested > 0:
        overall_color_acc = (total_color_correct / total_images_tested) * 100
        
        print(f"OVERALL ACCURACY: {overall_color_acc:.2f}%")
        print(f"Total Images: {total_images_tested}\n")
        
        if overall_color_acc >= 95.0:
            print("[PASSED] Targets Met")
        else:
            print("[NEEDS TUNING] Targets Not Met")
            
        df = pd.DataFrame(results_list)
        df.to_csv("accuracy_report.csv", index=False)
        print("Saved detailed report to 'accuracy_report.csv'")
    else:
        print("No images were tested.")

if __name__ == "__main__":
    evaluate_system()
