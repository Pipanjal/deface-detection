import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, LSTM, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import mixed_precision
from google.colab import drive

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)
base_path = '/content/drive/MyDrive/Celeb-DF'

# Utility to safely list directory contents
def safe_listdir(path, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return os.listdir(path)
        except Exception:
            time.sleep(delay)
    return []

# Custom data generator for sequence data
class DeepfakeDataGenerator(Sequence):
    def __init__(self, image_paths, batch_size=2, sequence_length=3, shuffle=True, augment=False, datagen=None):
        super().__init__()
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.augment = augment
        self.datagen = datagen
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / (self.batch_size * self.sequence_length)))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size * self.sequence_length:
                                       (idx + 1) * self.batch_size * self.sequence_length]
        batch_x, batch_y = [], []
        for i in range(0, len(batch_paths), self.sequence_length):
            sequence_x = []
            sequence_paths = batch_paths[i:i + self.sequence_length]
            if len(sequence_paths) < self.sequence_length:
                continue
            skip_seq = False
            for image_path, label in sequence_paths:
                try:
                    img = load_img(image_path, target_size=(128, 128))
                    img_array = img_to_array(img) / 255.0
                    if self.augment and self.datagen:
                        img_array = self.datagen.random_transform(img_array)
                    sequence_x.append(img_array)
                except Exception:
                    skip_seq = True
                    break
            if skip_seq:
                continue
            batch_x.append(np.array(sequence_x))
            batch_y.append(sequence_paths[0][1])
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)


def collect_image_paths(base_path):
    image_paths = []
    for category in ['celeb-real-output', 'celeb-fake-output', 'yt-output']:
        cat_dir = os.path.join(base_path, category)
        if not os.path.exists(cat_dir):
            continue
        images = safe_listdir(cat_dir)
        for image in images:
            if image.endswith(('.jpg', '.png')):
                label = 1 if category == 'celeb-fake-output' else 0
                image_paths.append((os.path.join(cat_dir, image), label))
    return image_paths


all_image_paths = collect_image_paths(base_path)
if len(all_image_paths) == 0:
    raise ValueError("No images found.")


num_fake = sum(label for _, label in all_image_paths)
num_real = len(all_image_paths) - num_fake
print(f"Label distribution: {num_fake} fake / {num_real} real")

np.random.shuffle(all_image_paths)
demo_size = 100
all_image_paths = all_image_paths[:demo_size]


sequence_length = 3
batch_size = 3
total_sequences = len(all_image_paths) // sequence_length
train_sequences = int(0.8 * total_sequences)
val_sequences = total_sequences - train_sequences

train_paths = all_image_paths[:train_sequences * sequence_length]
val_paths = all_image_paths[train_sequences * sequence_length:train_sequences * sequence_length + val_sequences * sequence_length]


datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


train_generator = DeepfakeDataGenerator(
    train_paths, batch_size=batch_size, sequence_length=sequence_length,
    shuffle=True, augment=True, datagen=datagen
)
val_generator = DeepfakeDataGenerator(
    val_paths, batch_size=batch_size, sequence_length=sequence_length,
    shuffle=False
)


checkpoint_path = os.path.join(base_path, 'demo_best_model_efficientnet_lstm.keras')
final_model_path = os.path.join(base_path, 'demo_deepfake_model_efficientnet_lstm_final.keras')


try:
    if os.path.exists(checkpoint_path):
        test_model = load_model(checkpoint_path)
        print("Checkpoint loaded successfully.")
        model = test_model
    else:
        raise FileNotFoundError
except Exception as e:
    print(f"Checkpoint corrupted or missing. Rebuilding model. Reason: {e}")
    efficientnet = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    for layer in efficientnet.layers[:-10]:
        layer.trainable = False
    model = Sequential([
        Input(shape=(sequence_length, 128, 128, 3)),
        TimeDistributed(efficientnet),
        TimeDistributed(Flatten()),
        LSTM(128, return_sequences=False),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid', dtype='float32')  
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

# Set up callbacks
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=1, restore_best_weights=True, verbose=1)
]

# Train the model
last_epoch = 0
model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    callbacks=callbacks,
    initial_epoch=last_epoch
)

# Save final model
model.save(final_model_path)

# Evaluate
val_loss, val_accuracy, val_auc = model.evaluate(val_generator)
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%, Validation AUC: {val_auc:.4f}")
