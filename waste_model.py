"""
BinSense – Waste CNN Classifier
TensorFlow / Keras – EfficientNetB0 Transfer Learning (Maximum Training)
"""

import numpy as np
import io
import os
from pathlib import Path
from typing import cast
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image


# ─── 7 Waste Categories (alphabetical folder/index order) ────────────────────
CATEGORIES = ['e_usable', 'general', 'glass', 'metal', 'organic', 'paper', 'plastic']

CATEGORY_INFO = {
    'e_usable': {'label': 'E-Usable (Electronics)',  'result': 'usable',     'bin': 'E-Waste Collection Center'},
    'general':  {'label': 'General Waste',           'result': 'waste',      'bin': 'General Waste Bin'},
    'glass':    {'label': 'Glass',                   'result': 'recyclable', 'bin': 'Glass Recycling Bin'},
    'metal':    {'label': 'Metal Waste',             'result': 'recyclable', 'bin': 'Metal Recycling Bin'},
    'organic':  {'label': 'Organic Waste',           'result': 'waste',      'bin': 'Compost / Green Bin'},
    'paper':    {'label': 'Paper / Cardboard',       'result': 'recyclable', 'bin': 'Paper Recycling Bin'},
    'plastic':  {'label': 'Plastic Waste',           'result': 'waste',      'bin': 'Waste Bin'},
}

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.50


# ══════════════════════════════════════════════════════════════════════════════
# CNN MODEL — EfficientNetB0 Transfer Learning
# ══════════════════════════════════════════════════════════════════════════════

class WasteClassifier:
    """
    Offline EfficientNetB0 classifier for the seven waste categories.

    Architecture:
        EfficientNetB0 (ImageNet weights, frozen for a fresh build)
        -> GlobalAveragePooling2D
        -> Dropout(0.3)
        -> Dense(7, softmax)
    """

    MODEL_PATH = os.environ.get('BINSENSE_MODEL_PATH', 'waste_cnn_model.keras')
    MODEL_CANDIDATES = ('waste_cnn_model.keras', 'waste_cnn_model.h5')

    def __init__(self) -> None:
        self.model: keras.Model = self._build_or_load()

    def _build_or_load(self) -> keras.Model:
        model_paths = [Path(self.MODEL_PATH)]
        model_paths.extend(Path(path) for path in self.MODEL_CANDIDATES)
        seen = set()
        for model_path in model_paths:
            if model_path in seen or not model_path.exists():
                continue
            seen.add(model_path)
            print(f"[CNN] Loading saved model from {model_path}")
            try:
                saved_model = cast(keras.Model, keras.models.load_model(model_path, compile=False))
                if self._has_requested_head(saved_model):
                    return saved_model
                print('[CNN] Saved model does not match the requested EfficientNet head; skipping it.')
            except Exception as exc:
                print(f"[CNN] Could not load {model_path}: {exc}")
        print("[CNN] Building EfficientNetB0 transfer learning model...")
        return self._build_model()

    @staticmethod
    def _has_requested_head(model: keras.Model) -> bool:
        """Ensure a checkpoint has the exact seven-class prediction head."""
        if model.output_shape[-1] != len(CATEGORIES):
            return False
        layers = model.layers
        if len(layers) < 3:
            return False
        pooling = layers[-3]
        dropout = layers[-2]
        output = layers[-1]
        return (
            isinstance(pooling, keras.layers.GlobalAveragePooling2D)
            and isinstance(dropout, keras.layers.Dropout)
            and abs(float(dropout.rate) - 0.3) < 1e-6
            and isinstance(output, keras.layers.Dense)
            and output.units == len(CATEGORIES)
            and output.activation.__name__ == 'softmax'
        )

    def _build_model(self) -> keras.Model:
        # ── Backbone: EfficientNetB0 pretrained on ImageNet ───────────────────
        # EfficientNetB0 handles its own preprocessing internally ([0,255] input)
        base_model = keras.applications.EfficientNetB0(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights='imagenet',
        )
        base_model.trainable = False

        # ── Custom Classification Head ─────────────────────────────────────
        inputs = keras.Input(shape=(*IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)

        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(len(CATEGORIES), activation='softmax')(x)

        model = keras.Model(inputs, outputs, name='WasteCNN_EfficientNetB0')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
        )

        print(f"[CNN] Model built. Parameters: {model.count_params():,}")
        return model

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert raw image bytes → float32 array (1, 224, 224, 3).
        Resize to 224x224 and apply EfficientNet preprocessing.
        Using LANCZOS for high-quality resizing to improve thin item detection.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        arr = self.preprocess(image_bytes)
        predictions = self.model.predict(arr, verbose=0)[0]

        class_idx  = int(np.argmax(predictions))
        confidence = float(predictions[class_idx]) * 100
        category   = CATEGORIES[class_idx]
        print(f"[CNN] Predicted {category} with {confidence:.1f}% confidence")

        # ── Threshold Logic: If confidence < 50%, force 'general' ──────────
        # This handles unlisted items (like wood) or highly uncertain items.
        if confidence < CONFIDENCE_THRESHOLD * 100:
            print(f"[CNN] Low confidence ({confidence:.1f}%); falling back to 'general'")
            category = 'general'
            # Note: We keep the original confidence but change the category

        info = CATEGORY_INFO[category]

        top3_idx = np.argsort(predictions)[::-1][:3]
        top3 = [
            {
                'category':   CATEGORIES[i],
                'label':      CATEGORY_INFO[CATEGORIES[i]]['label'],
                'confidence': round(float(predictions[i]) * 100, 1),
            }
            for i in top3_idx
        ]

        return {
            'category':   category,
            'label':      info['label'],
            'result':     info['result'],
            'bin':        info['bin'],
            'confidence': round(confidence, 1),
            'top3':       top3,
        }

    def train(self, train_dataset, val_dataset,
              epochs: int = 50, fine_tune: bool = True,
              class_weight: dict | None = None):
        """
        Phase 1 — train head only (backbone frozen).
        Phase 2 — unfreeze last 50 EfficientNetB0 layers and fine-tune.
        """
        print("\n[CNN] Phase 1: Training classification head...")
        self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            class_weight=class_weight,
            callbacks=self._callbacks('phase1'),
        )

        if fine_tune:
            print("\n[CNN] Phase 2: Fine-tuning last 50 backbone layers...")

            # Find EfficientNetB0 base layer
            base_model = None
            for layer in self.model.layers:
                if 'efficientnet' in layer.name.lower():
                    base_model = layer
                    break

            if base_model is None:
                print("[CNN] WARNING: backbone layer not found — skipping fine-tune")
            else:
                base_model.trainable = True
                for layer in base_model.layers[:-50]:
                    layer.trainable = False

                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=5e-6),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                )

                self.model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=25,
                    class_weight=class_weight,
                    callbacks=self._callbacks('phase2'),
                )

        self.model.save(self.MODEL_PATH)
        print(f"[CNN] Model saved to {self.MODEL_PATH}")

    def _callbacks(self, name: str = '') -> list:
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=8,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.3,
                patience=4, min_lr=1e-8, verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                f'best_model_{name}.keras',
                save_best_only=True, monitor='val_accuracy', verbose=0,
            ),
        ]

    def evaluate(self, test_dataset) -> list:
        results = self.model.evaluate(test_dataset, verbose='auto')
        print(f"\n[CNN] Validation Accuracy: {results[1]*100:.2f}%")
        return results

    def summary(self):
        self.model.summary()


# ══════════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def get_augmentation_layer() -> keras.Sequential:
    """Heavy augmentation pipeline — maximizes generalization on small dataset."""
    return keras.Sequential([
        keras.layers.RandomFlip('horizontal_and_vertical'),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.2),
    ], name='augmentation')
