import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
try:
    from matplotlib.patches import Rectangle
except ImportError:
    Rectangle = None
import xml.etree.ElementTree as ET

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset-iiit-pet')
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, 'annotations/xmls')

def get_image_and_box(filename):
    """
    Laad een afbeelding en de bounding box uit de Oxford-IIIT Pet dataset.
    Sla afbeeldingen zonder object of bounding box over.
    """
    image_path = os.path.join(IMAGES_PATH, filename + '.jpg')
    annot_path = os.path.join(ANNOTATIONS_PATH, filename + '.xml')
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        tree = ET.parse(annot_path)
        root = tree.getroot()
        obj = root.find('object')
        if obj is None:
            return None, None, None
        label_elem = obj.find('name')
        bndbox = obj.find('bndbox')
        if label_elem is None or bndbox is None:
            return None, None, None
        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')
        values = [xmin, ymin, xmax, ymax]
        texts = []
        for v in values:
            if v is None or v.text is None or not isinstance(v.text, str) or v.text.strip() == '':
                return None, None, None
            texts.append(v.text.strip())
        try:
            box = [int(t) for t in texts]
        except (TypeError, ValueError):
            return None, None, None
        label = label_elem.text if label_elem is not None else 'unknown'
        return img, box, label
    except Exception:
        return None, None, None

def plot_image_with_box(img, box, label):
    """ Plot de afbeelding met de bounding box en label."""
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    x1, y1, x2, y2 = box
    if Rectangle is not None:
        plt.gca().add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        )
    plt.text(x1, y1, label, color='white', bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Zoek het eerste geldige voorbeeld
    for fname in os.listdir(IMAGES_PATH):
        if not fname.endswith('.jpg'):
            continue
        filebase = fname.replace('.jpg', '')
        img, box, label = get_image_and_box(filebase)
        print(filebase)
        # print(f"Processing {filebase}: img={img is not None}, box={box}, label={label}")
        if img is not None and box is not None and label is not None:
            plot_image_with_box(img, box, label)
            break