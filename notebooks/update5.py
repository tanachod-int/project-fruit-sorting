import json

notebook_path = r'd:\Folder Top\Homework\Senior\Term2\CS483\project-fruit-sorting\notebooks\Project_Fruit_Sorting.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = [
    {
      'cell_type': 'markdown',
      'metadata': {},
      'source': [
        '## Section 5: Step 5 — Classification & Results\n',
        'การใช้กฎ (Rule-based) เพื่อตัดสินว่าวัตถุที่ตรวจจับได้คือผลไม้ชนิดไหน และขนาดเท่าไหร่'
      ]
    },
    {
      'cell_type': 'code',
      'execution_count': None,
      'metadata': {},
      'outputs': [],
      'source': [
        'def classify_fruit(features: dict) -> dict:\n',
        '    \"\"\"Classifies fruit based on hue and size (area).\"\"\"\n',
        '    hue = features[\'mean_hue\']\n',
        '    area = features[\'area\']\n',
        '    \n',
        '    # 1. จำแนกสี (Color Class)\n',
        '    if (0 <= hue <= 12) or (165 <= hue <= 180):\n',
        '        color_label = "Red (Strawberry/Apple)"\n',
        '    elif 13 <= hue <= 25:\n',
        '        color_label = "Orange (Orange/Mango)"\n',
        '    elif 26 <= hue <= 38:\n',
        '        color_label = "Yellow (Banana/Mango)"\n',
        '    elif 39 <= hue <= 90:\n',
        '        color_label = "Green (Green Grape/Mango)"\n',
        '    else:\n',
        '        color_label = "Unknown"\n',
        '\n',
        '    # 2. จำแนกขนาด (Size Class) - ปรับ Threshold ตามข้อมูลตัวอย่าง\n',
        '    if area < 5000:\n',
        '        size_label = "Small"\n',
        '    elif area < 15000:\n',
        '        size_label = "Medium"\n',
        '    else:\n',
        '        size_label = "Large"\n',
        '        \n',
        '    return {"color": color_label, "size": size_label}'
      ]
    },
    {
      'cell_type': 'code',
      'execution_count': None,
      'metadata': {},
      'outputs': [],
      'source': [
        'if original_img is not None and len(fruit_contours) > 0:\n',
        '    print("--- Final Classification Results ---")\n',
        '    result_img = original_img.copy()\n',
        '    \n',
        '    for i, contour in enumerate(fruit_contours):\n',
        '        features = extract_features(contour, original_img)\n',
        '        prediction = classify_fruit(features)\n',
        '        \n',
        '        # แสดงผลลัพธ์ทางหน้าจอ\n',
        '        print(f"Fruit #{i+1}: {prediction[\'color\']} | Size: {prediction[\'size\']}")\n',
        '        \n',
        '        # วาดข้อความทับลงบรูปภาพ\n',
        '        x, y, w, h = features[\'bounding_box\']\n',
        '        text = f"#{i+1}: {prediction[\'size\']}"\n',
        '        cv2.putText(result_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)\n',
        '        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)\n',
        '\n',
        '    show_comparison(original_img, result_img, "Original", "Classified Output")'
      ]
    }
]

nb['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
