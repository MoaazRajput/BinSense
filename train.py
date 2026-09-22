"""
BinSense — Maximum Training Script
Run: python train.py

Backbone  : EfficientNetB0 (ImageNet weights)
Phase 1   : 50 epochs, head only, batch=16, class weights
Phase 2   : 25 epochs, last 50 layers unfrozen, LR=5e-6
"""

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras

from waste_model import (
    WasteClassifier,
    CATEGORIES,
    IMG_SIZE,
    get_augmentation_layer,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = 'dataset'
BATCH_SIZE = 16       # smaller batch → more gradient updates → better generalization
EPOCHS     = 50       # EarlyStopping will stop early if needed
VAL_SPLIT  = 0.2
IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# ── Header ────────────────────────────────────────────────────────────────────
print('\n' + '=' * 55)
print('  BinSense — Maximum Training  (EfficientNetB0)')
print('=' * 55)
print(f'\n  TensorFlow : {tf.__version__}')
gpu_count = len(tf.config.list_physical_devices('GPU'))
print(f'  GPU        : {"Available (" + str(gpu_count) + ")" if gpu_count else "Not found — using CPU"}')

# ── Dataset check ─────────────────────────────────────────────────────────────
if not os.path.exists(DATA_DIR):
    print(f'\n  ERROR: dataset/ folder not found.')
    print('  Run first:  python organize_dataset.py')
    sys.exit(1)

print(f'\n  Dataset check ({DATA_DIR}/):\n')

total_images  = 0
cat_counts    = {}
empty_cats    = []
low_cats      = []

for cat in CATEGORIES:
    cat_path = os.path.join(DATA_DIR, cat)
    count = (
        sum(1 for f in os.listdir(cat_path)
            if os.path.splitext(f)[1].lower() in IMG_EXTS)
        if os.path.exists(cat_path) else 0
    )
    cat_counts[cat] = count
    total_images   += count
    icon = 'OK' if count >= 50 else ('!!' if count > 0 else 'XX')
    print(f'    [{icon}]  {count:>4} images  ->  {DATA_DIR}/{cat}/')

    if count == 0:
        empty_cats.append(cat)
    elif count < 50:
        low_cats.append((cat, count))

print(f'\n  Total: {total_images} images across {len(CATEGORIES)} categories')

if empty_cats:
    print(f'\n  ERROR: No images in: {", ".join(empty_cats)}')
    sys.exit(1)

# ── Class weights (fix imbalanced dataset) ────────────────────────────────────
# Gives more weight to under-represented categories (e.g. plastic=88 vs mouse=197)
class_weight = {
    i: total_images / (len(CATEGORIES) * cat_counts[cat])
    for i, cat in enumerate(CATEGORIES)
}
print('\n  Class weights (imbalance correction):')
for i, cat in enumerate(CATEGORIES):
    print(f'    {cat:<22} {class_weight[i]:.3f}')

# ── Load datasets ─────────────────────────────────────────────────────────────
print('\n  Loading dataset...')

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset='training',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=CATEGORIES,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset='validation',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=CATEGORIES,
)

# ── Augmentation (training only) ──────────────────────────────────────────────
augment  = get_augmentation_layer()
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    lambda x, y: (augment(x, training=True), y),
    num_parallel_calls=AUTOTUNE,
).prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)

train_batches = tf.data.experimental.cardinality(train_ds).numpy()
val_batches   = tf.data.experimental.cardinality(val_ds).numpy()
print(f'  Train batches: {train_batches}  |  Val batches: {val_batches}')

# ── Build & train ─────────────────────────────────────────────────────────────
print('\n  Building model...')
clf = WasteClassifier()
clf.summary()

print(f'\n  Starting training')
print(f'  Batch size : {BATCH_SIZE}')
print(f'  Phase 1    : up to {EPOCHS} epochs  (head only, EarlyStopping patience=8)')
print(f'  Phase 2    : up to 25 epochs  (last 50 layers, LR=5e-6)\n')

clf.train(
    train_ds, val_ds,
    epochs=EPOCHS,
    fine_tune=True,
    class_weight=class_weight,
)

# ── Final evaluation ──────────────────────────────────────────────────────────
print('\n' + '=' * 55)
print('  Final Evaluation on Validation Set')
print('=' * 55)

results = clf.evaluate(val_ds)
acc = results[1] * 100

print(f'\n  Accuracy : {acc:.2f}%')
print(f'  Model    : waste_cnn_model.keras')
print('\n  Start server:  python app.py')
print('  Open:          http://localhost:5000\n')
