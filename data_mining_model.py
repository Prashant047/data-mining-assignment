import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


DATA_PATH = "./kaggleEyePACSDataset"
TRAIN_LABELS = os.path.join(DATA_PATH, "trainLabels.csv")
TRAIN_DATA = os.path.join(DATA_PATH, "train")

labels = pd.read_csv(TRAIN_LABELS)
labels['image_path'] = labels.apply(lambda row: os.path.join(TRAIN_DATA, str(row['image']) + '.jpeg'), axis=1)

def load_image(image_path):
    return cv2.imread(image_path)

def clean_image(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def augment_image(image):

    alpha = 1 + np.random.uniform(-0.1, 0.1)
    beta = np.random.uniform(-20, 20)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image

def normalize_and_resize(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image

def preprocess_image(image_path):
    image = load_image(image_path)
    image = clean_image(image)
    image = augment_image(image)

    # Applying CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    image = normalize_and_resize(image)
    return image

labels['image'] = labels['image_path'].apply(preprocess_image)

X = np.stack(labels['image'].to_numpy())
y = labels['level'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


data_gen = ImageDataGenerator(rotation_range=20,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True,
                              vertical_flip=True)

# Model Creation
def create_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

model = create_model()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Training
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10)

history = model.fit(data_gen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=100,
                    callbacks=[checkpoint, early_stopping])

# Save the final model
model.save('final_model.h5')
