import os
import numpy as np
import cv2
import tensorflow as tf
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset-iiit-pet')
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, 'annotations/xmls')

IMG_SIZE = 256  # Klein houden voor snelheid


def get_label_mapping():
    """Laad label mapping van de dataset."""
    all_labels = set()
    for fname in os.listdir(ANNOTATIONS_PATH):
        if not fname.endswith('.xml'):
            continue
        tree = ET.parse(os.path.join(ANNOTATIONS_PATH, fname))
        root = tree.getroot()
        obj = root.find('object')
        if obj is not None:
            label_elem = obj.find('name')
            if label_elem is not None:
                all_labels.add(label_elem.text)
    return {label: i for i, label in enumerate(sorted(all_labels))}

def load_data(label2idx, img_size=128):
    """Laad afbeeldingen en bounding boxes uit de dataset."""
    X, Y_box, Y_label = [], [], []
    for fname in os.listdir(IMAGES_PATH):
        if not fname.endswith('.jpg'):
            continue
        filebase = fname.replace('.jpg', '')
        image_path = os.path.join(IMAGES_PATH, fname)
        annot_path = os.path.join(ANNOTATIONS_PATH, filebase + '.xml')
        if not os.path.exists(annot_path):
            continue
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tree = ET.parse(annot_path)
        root = tree.getroot()
        obj = root.find('object')
        if obj is None:
            continue
        label_elem = obj.find('name')
        bndbox = obj.find('bndbox')
        if label_elem is None or bndbox is None:
            continue
        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')
        values = [xmin, ymin, xmax, ymax]
        texts = []
        for v in values:
            if v is None or v.text is None or not isinstance(v.text, str) or v.text.strip() == '':
                break
            texts.append(v.text.strip())
        if len(texts) != 4:
            continue
        try:
            box = [int(t) for t in texts]
        except (TypeError, ValueError):
            continue
        h, w = img.shape[:2]
        box_norm = [box[0]/w, box[1]/h, box[2]/w, box[3]/h]
        img_resized = cv2.resize(img, (img_size, img_size))
        X.append(img_resized)
        Y_box.append(box_norm)
        Y_label.append(label2idx[label_elem.text])
    X = np.array(X, dtype=np.float32) / 255.0
    Y_box = np.array(Y_box, dtype=np.float32)
    Y_label = np.array(Y_label, dtype=np.int32)
    return X, Y_box, Y_label

def build_model(num_classes, img_size=128):
    """Bouw een eenvoudig CNN model voor object detectie."""

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    box_out = tf.keras.layers.Dense(4, activation='sigmoid', name='box')(x)
    class_out = tf.keras.layers.Dense(num_classes, activation='softmax', name='label')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[box_out, class_out])
    model.compile(
        optimizer='adam',
        loss={'box': 'mean_squared_error', 'label': 'sparse_categorical_crossentropy'},
        metrics={'box': 'mean_squared_error', 'label': 'accuracy'}
)
    return model


def main():
    label2idx = get_label_mapping()
    print(f"Aantal klassen: {len(label2idx)}")
    image_sizes = [64, 128, 224]  # Experimenteer met verschillende resoluties
    results = []
    for img_size in image_sizes:
        print(f"\n==== Training met image size: {img_size} ====")
        X, Y_box, Y_label = load_data(label2idx, img_size)
        print(f"Aantal samples: {len(X)}")
        X_train, X_test, Y_box_train, Y_box_test, Y_label_train, Y_label_test = train_test_split(
            X, Y_box, Y_label, test_size=0.2, random_state=42)
        model = build_model(len(label2idx), img_size)
        model.summary()
        
        # Early stopping om overfitting te voorkomen
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        history = model.fit(
            X_train, {'box': Y_box_train, 'label': Y_label_train},
            validation_data=(X_test, {'box': Y_box_test, 'label': Y_label_test}),
            epochs=100, batch_size=32, verbose=2,
            callbacks=[early_stopping]
        )
        # Plot accuracy per epoch
        f, axarr = plt.subplots(1, 2, figsize=(12, 5))
        axarr[0].plot(history.history['label_accuracy'], label='Train acc')
        axarr[0].plot(history.history['val_label_accuracy'], label='Val acc')
        axarr[0].set_title(f'Accuracy per epoch (img_size={img_size})')
        axarr[0].set_xlabel('Epoch')
        axarr[0].set_ylabel('Accuracy')
        axarr[0].legend()

        axarr[1].plot(history.history['box_mean_squared_error'], label='Train box MSE')
        axarr[1].plot(history.history['val_box_mean_squared_error'], label='Val box MSE')
        axarr[1].set_title(f'Box MSE per epoch (img_size={img_size})')
        axarr[1].set_xlabel('Epoch')
        axarr[1].set_ylabel('Box MSE')
        axarr[1].legend()
        plt.savefig(f'accuracy_curve_{img_size}.png')
        plt.close()

        
        loss, box_loss, label_loss, box_mse, label_acc = model.evaluate(
            X_test, {'box': Y_box_test, 'label': Y_label_test}, verbose=0
        )
        print(f"Test box MSE: {box_mse:.4f}, Test label acc: {label_acc:.4f}")
        model.save(f'models/pet_detector_model_{img_size}.keras')
        print(f'Model opgeslagen als pet_detector_model_{img_size}.keras')
        results.append({'img_size': img_size, 'box_mse': box_mse, 'label_acc': label_acc})

    print("\nVergelijking resultaten:")
    for res in results:
        print(f"Image size: {res['img_size']}, Box MSE: {res['box_mse']:.4f}, Label acc: {res['label_acc']:.4f}")

if __name__ == '__main__':
    main()

