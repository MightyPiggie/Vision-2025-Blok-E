import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

from train_pet_detector import get_label_mapping, load_data


# Verschillende modelconfiguraties om te testen
layer_configs = [
    # config 1: klein model, relu, alleen Conv2D
    {'name': 'small_relu', 'layers': [
        {'type': 'conv', 'filters': 32, 'activation': 'relu'},
        {'type': 'conv', 'filters': 64, 'activation': 'relu'}
    ]},
    # config 2: medium model, leakyrelu, Conv2D + BatchNorm
    {'name': 'medium_leakyrelu_bn', 'layers': [
        {'type': 'conv', 'filters': 32, 'activation': 'leakyrelu'},
        {'type': 'batchnorm'},
        {'type': 'conv', 'filters': 64, 'activation': 'leakyrelu'},
        {'type': 'batchnorm'},
        {'type': 'conv', 'filters': 128, 'activation': 'leakyrelu'}
    ]},
    # config 3: groot model, elu, Conv2D + Dropout
    {'name': 'large_elu_dropout', 'layers': [
        {'type': 'conv', 'filters': 32, 'activation': 'elu'},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'conv', 'filters': 64, 'activation': 'elu'},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'conv', 'filters': 128, 'activation': 'elu'},
        {'type': 'conv', 'filters': 256, 'activation': 'elu'}
    ]},
]

IMG_SIZE = 128
EPOCHS = 100
BATCH_SIZE = 32

label2idx = get_label_mapping()
X, Y_box, Y_label = load_data(label2idx, IMG_SIZE)
X_train, X_test, Y_box_train, Y_box_test, Y_label_train, Y_label_test = train_test_split(
    X, Y_box, Y_label, test_size=0.2, random_state=42)

results = []

for config in layer_configs:
    print(f"\n==== Training model: {config['name']} ====")
    # Dynamisch model bouwen
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    for layer in config['layers']:
        if layer['type'] == 'conv':
            act = layer['activation']
            if act == 'leakyrelu':
                x = tf.keras.layers.Conv2D(layer['filters'], 3, padding='same')(x)
                x = tf.keras.layers.LeakyReLU()(x)
            else:
                x = tf.keras.layers.Conv2D(layer['filters'], 3, activation=act, padding='same')(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        elif layer['type'] == 'batchnorm':
            x = tf.keras.layers.BatchNormalization()(x)
        elif layer['type'] == 'dropout':
            x = tf.keras.layers.Dropout(layer['rate'])(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    box_out = tf.keras.layers.Dense(4, activation='sigmoid', name='box')(x)
    class_out = tf.keras.layers.Dense(len(label2idx), activation='softmax', name='label')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[box_out, class_out])
    model.compile(
        optimizer='adam',
        loss={'box': 'mean_squared_error', 'label': 'sparse_categorical_crossentropy'},
        metrics={'box': 'mean_squared_error', 'label': 'accuracy'}
    )
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
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
        callbacks=[early_stopping]
    )
    loss, box_loss, label_loss, box_mse, label_acc = model.evaluate(
        X_test, {'box': Y_box_test, 'label': Y_label_test}, verbose=0
    )
    print(f"Test box MSE: {box_mse:.4f}, Test label acc: {label_acc:.4f}")
    model.save(f'pet_detector_{config["name"]}.keras')
    results.append({'name': config['name'], 'box_mse': box_mse, 'label_acc': label_acc, 'history': history})

# Plot accuracies per epoch
plt.figure(figsize=(10,6))
for res in results:
    plt.plot(res['history'].history['val_label_accuracy'], label=f"{res['name']} (val)")
plt.title('Validation accuracy per epoch per model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('compare_model_accuracies.png')
plt.show()

# Print samenvatting
print("\nVergelijking resultaten:")
for res in results:
    print(f"Model: {res['name']}, Box MSE: {res['box_mse']:.4f}, Label acc: {res['label_acc']:.4f}")
