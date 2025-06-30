import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D,
    Concatenate, BatchNormalization, Dropout
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical, Sequence
import pickle

# --- STEP 1: CHECK GOOGLE DRIVE MOUNTING (if using Google Colab) ---
# If running in Google Colab, uncomment and mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# --- STEP 2: LOCAL ENVIRONMENT FILE PATHS ---
# Modify these paths based on your local file structure
image_folder = r"D:\internship\download.jpg"  # Raw string for avoiding escape issues
csv_path = r"D:\internship\my_file.csv"  # Update to your local CSV path

# Print absolute path and check if the CSV file exists
print(f"Trying to load CSV from: {os.path.abspath(csv_path)}")

if not os.path.exists(csv_path):
    print(f"Error: The file '{csv_path}' does not exist.")
    exit()
else:
    print(f"File exists: {csv_path}")

# --- STEP 3: LOAD METADATA & COLLECT IMAGE PATHS ---
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: The file '{csv_path}' was not found.")
    exit()

# Add image_path column
df['image_path'] = df['filename'].apply(lambda x: os.path.join(image_folder, x))
df['full_path'] = df['image_path']

# Check if any image files are missing
missing = df[~df['image_path'].apply(os.path.exists)]
if len(missing) > 0:
    print(f"⚠️ Missing image files: {len(missing)}")
    print("Missing files:", missing[['filename', 'image_path']])
else:
    print(f"🔍 Found {df['image_path'].nunique()} image files.")

# --- STEP 4: PREPROCESSING ---
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)

meta_columns = ['symptom_lethargy', 'symptom_diarrhea', 'egg_production_rate', 'temperature', 'humidity']
scaler = StandardScaler()
df[meta_columns] = scaler.fit_transform(df[meta_columns])

# Train/Val split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_enc'], random_state=42)

# --- STEP 5: CUSTOM GENERATOR ---
class MultiInputGenerator(Sequence):
    def __init__(self, dataframe, meta_columns, num_classes, batch_size=32, img_size=(224, 224), shuffle=True):
        self.df = dataframe.reset_index(drop=True)
        self.meta_columns = meta_columns
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_img = np.array([
            img_to_array(load_img(path, target_size=self.img_size)) / 255.0
            for path in batch['full_path']
        ], dtype=np.float32)

        X_meta = batch[self.meta_columns].values.astype(np.float32)
        y = to_categorical(batch['label_enc'], num_classes=self.num_classes)
        return (X_img, X_meta), y

# --- STEP 6: MODEL DEFINITION ---
# Image branch
img_input = Input(shape=(224, 224, 3))
base_model = MobileNetV2(include_top=False, input_tensor=img_input, weights='imagenet')
base_model.trainable = False
x_img = GlobalAveragePooling2D()(base_model.output)

# Meta branch
meta_input = Input(shape=(len(meta_columns),))
x_meta = Dense(64, activation='relu')(meta_input)
x_meta = BatchNormalization()(x_meta)

# Fusion
combined = Concatenate()([x_img, x_meta])
x = Dense(128, activation='relu')(combined)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=[img_input, meta_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- STEP 7: TRAINING ---
train_gen = MultiInputGenerator(train_df, meta_columns, num_classes, batch_size=16)
val_gen = MultiInputGenerator(val_df, meta_columns, num_classes, batch_size=16, shuffle=False)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

loss, acc = model.evaluate(val_gen, verbose=0)
print(f"✅ Validation Accuracy: {acc:.4f}")

# --- STEP 8: SAVE MODEL ---
model.save("multimodal_poultry_model.h5")

# Try to convert the model to TFLite format
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("multimodal_poultry_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ TFLite model saved successfully")
except Exception as e:
    print(f"Error during TFLite conversion: {str(e)}")

# Optionally save LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model and LabelEncoder saved as .h5, .tflite, and .pkl")

# --- STEP 9: VALIDATE IMAGE BATCH SHAPE ---
# Print batch shape to ensure correct data is being passed to the model
batch = train_gen.__getitem__(0)
X_img, X_meta = batch[0]
y = batch[1]
print(f"Image batch shape: {X_img.shape}")
print(f"Metadata batch shape: {X_meta.shape}")
print(f"Labels batch shape: {y.shape}")

# --- STEP 10: DEBUG IMAGE LOADING ---
# Check a sample image shape to ensure it is loaded correctly
sample_image_path = df['full_path'].iloc[0]
img = load_img(sample_image_path, target_size=(224, 224))
img_array = img_to_array(img)
print(f"Sample image shape: {img_array.shape}")
