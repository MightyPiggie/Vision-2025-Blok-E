import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xml.etree.ElementTree as ET

# Laad label mapping
from train_pet_detector import get_label_mapping, IMG_SIZE, DATASET_PATH, IMAGES_PATH, ANNOTATIONS_PATH
label2idx = get_label_mapping()
idx2label = {v: k for k, v in label2idx.items()}

# Laad model
model = tf.keras.models.load_model('pet_detector_model.keras')

def parse_ground_truth(annot_path):
    try:
        tree = ET.parse(annot_path)
        root = tree.getroot()
        obj = root.find('object')
        if obj is None:
            return 0, 0, 0, 0, 'unknown'
        label_elem = obj.find('name')
        bndbox = obj.find('bndbox')
        if label_elem is None or bndbox is None:
            return 0, 0, 0, 0, 'unknown'
        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')
        values = [xmin, ymin, xmax, ymax]
        texts = []
        for v in values:
            if v is None or v.text is None or not isinstance(v.text, str) or v.text.strip() == '':
                return 0, 0, 0, 0, 'unknown'
            texts.append(v.text.strip())
        if len(texts) != 4:
            return 0, 0, 0, 0, 'unknown'
        try:
            box = [int(t) for t in texts]
        except (TypeError, ValueError):
            return 0, 0, 0, 0, 'unknown'
        label = label_elem.text if label_elem is not None else 'unknown'
        return box[0], box[1], box[2], box[3], label
    except Exception:
        return 0, 0, 0, 0, 'unknown'


all_images = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg')]
all_images.sort()
for img_name in all_images:
    filebase = img_name.replace('.jpg', '')
    img_path = os.path.join(IMAGES_PATH, img_name)
    annot_path = os.path.join(ANNOTATIONS_PATH, filebase + '.xml')
    if not os.path.exists(annot_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    input_img = np.expand_dims(img_resized / 255.0, axis=0)
    pred_box, pred_label = model.predict(input_img)
    pred_box = pred_box[0]
    pred_label_idx = np.argmax(pred_label[0])
    pred_label_name = idx2label[pred_label_idx]
    x1 = int(pred_box[0] * w)
    y1 = int(pred_box[1] * h)
    x2 = int(pred_box[2] * w)
    y2 = int(pred_box[3] * h)
    xmin, ymin, xmax, ymax, gt_label = parse_ground_truth(annot_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.gca().add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='lime', linewidth=2, label='Ground Truth'))
    plt.text(xmin, ymin-10, f"GT: {gt_label}", color='lime', bbox=dict(facecolor='black', alpha=0.5))
    plt.gca().add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2, label='Prediction'))
    plt.text(x1, y1-10, f"Pred: {pred_label_name}", color='red', bbox=dict(facecolor='black', alpha=0.5))
    plt.axis('off')
    plt.legend(handles=[Rectangle((0,0),1,1,fill=False,color='lime',label='Ground Truth'), Rectangle((0,0),1,1,fill=False,color='red',label='Prediction')])
    plt.title(f"{img_name}")
    plt.show()
    input('Druk op Enter voor de volgende afbeelding...')
