"""
HeyDittoNet model architecture for wake word detection.

This module defines HeyDittoNet v3, a pure CNN architecture with
Squeeze-and-Excitation attention blocks optimized for keyword spotting.

Model features:
- Depthwise Separable Convolutions (MobileNet-style efficiency)
- Squeeze-and-Excitation blocks (channel attention)
- Progressive dropout (0.1 -> 0.5)
- Global Average Pooling (replaces LSTM/flatten)
- L2 regularization throughout
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from typing import Tuple, Optional


def se_block(x: tf.Tensor, reduction: int = 8) -> tf.Tensor:
    """
    Squeeze-and-Excitation block for channel attention.

    Learns to weight channels by their importance, effectively
    learning which frequency bands are most relevant for the wake word.
    """
    channels = K.int_shape(x)[-1]

    se = layers.GlobalAveragePooling2D()(x)

    se = layers.Dense(
        max(channels // reduction, 8),
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(se)
    se = layers.Dense(channels, activation='sigmoid')(se)

    se = layers.Reshape((1, 1, channels))(se)

    return layers.Multiply()([x, se])


def ds_conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    use_se: bool = True,
    dropout_rate: float = 0.1
) -> tf.Tensor:
    """Depthwise Separable Convolution block with optional SE attention."""
    # Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding='same',
        depthwise_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Pointwise convolution
    x = layers.Conv2D(
        filters,
        (1, 1),
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    if use_se:
        x = se_block(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    use_se: bool = True,
    dropout_rate: float = 0.1
) -> tf.Tensor:
    """Standard convolution block with optional SE attention."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    if use_se:
        x = se_block(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


def create_heydittonet_v3(input_shape: Tuple[int, int, int]) -> Model:
    """
    Create HeyDittoNet v3 model.

    Architecture:
        Input -> Resize -> Normalize ->
        Conv Block (32 filters) -> MaxPool ->
        DS Conv Block (48 filters) + SE -> MaxPool ->
        DS Conv Block (64 filters) + SE -> MaxPool ->
        DS Conv Block (96 filters) + SE -> MaxPool ->
        DS Conv Block (128 filters) + SE ->
        Global Average Pooling ->
        Dense (96) -> Dense (48) -> Dense (1, sigmoid)

    Args:
        input_shape: Shape of input spectrogram (height, width, channels)
                    Typically (149, 32, 1) for 1.5-second logfbank spectrogram

    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Resizing(40, 40)(inputs)
    x = layers.Normalization()(x)

    # Initial Conv Block
    x = conv_block(x, filters=32, kernel_size=3, strides=1, use_se=False, dropout_rate=0.0)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 40 -> 20

    # Block 1
    x = ds_conv_block(x, filters=48, use_se=True, dropout_rate=0.1)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 20 -> 10

    # Block 2
    x = ds_conv_block(x, filters=64, use_se=True, dropout_rate=0.15)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 10 -> 5

    # Block 3
    x = ds_conv_block(x, filters=96, use_se=True, dropout_rate=0.2)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)  # 5 -> 3

    # Block 4
    x = ds_conv_block(x, filters=128, use_se=True, dropout_rate=0.25)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense classifier head
    x = layers.Dense(
        96,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(
        48,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='HeyDittoNet_v3')

    return model


def compile_model(
    model: Model,
    learning_rate: float = 0.001
) -> Model:
    """Compile the model with optimizer and metrics."""
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model


def get_callbacks(
    patience: int = 15,
    min_delta: float = 0.001
) -> list:
    """Get training callbacks for early stopping and learning rate scheduling."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            min_delta=min_delta
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    return callbacks


def load_model(model_path: str) -> Model:
    """Load a saved model."""
    return keras.models.load_model(model_path)


def convert_to_tflite(
    model: Model,
    output_path: str,
    quantize: bool = False
) -> None:
    """Convert model to TensorFlow Lite format for edge deployment."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")


def print_model_summary(input_shape: Tuple[int, int, int] = (149, 32, 1)) -> None:
    """Create and print model summary."""
    model = create_heydittonet_v3(input_shape)
    model = compile_model(model)
    model.summary()

    return model


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HeyDittoNet v3 - Model Architecture")
    print("="*60 + "\n")

    model = print_model_summary()
