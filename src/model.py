"""Model architectures for OvaScope.

Both models take raw RGB pixels in [0, 255] (shape: image_size x image_size x 3)
and rescale *inside* the network. Keeping normalisation in the model means the
saved file carries its own preprocessing, so the notebook, API and retraining
cannot accidentally normalise twice or forget to normalise.
"""
from __future__ import annotations

import tensorflow as tf

layers = tf.keras.layers
BACKBONE_LAYER_NAME = "mobilenetv2_backbone"


def build_baseline_cnn(image_size: int, num_classes: int, dropout: float = 0.3) -> tf.keras.Model:
    """Small custom CNN: 4 x (Conv -> ReLU -> MaxPool), global average pooling, softmax.

    Filters double each block (32 -> 256) while the image shrinks, so the network
    learns simple edges first and more complex shell/texture patterns later.
    Batch normalisation after each convolution keeps training stable at a
    normal learning rate. Global average pooling (instead of Flatten) keeps the
    parameter count small, which limits overfitting on ~9k training images.
    """
    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale_0_1")(inputs)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="ovascope_baseline_cnn")


def build_transfer_model(image_size: int, num_classes: int, dropout: float = 0.3,
                         weights: str | None = "imagenet") -> tf.keras.Model:
    """MobileNetV2 backbone (ImageNet weights) + custom classification head.

    * Rescaling(1/127.5, offset=-1) maps [0,255] -> [-1,1], exactly what
      tf.keras.applications.mobilenet_v2.preprocess_input does.
    * The backbone is called with training=False so its BatchNormalization
      layers keep their ImageNet statistics, even after we unfreeze layers for
      fine-tuning (updating BN statistics on small batches destroys pretrained features).
    * Head: global average pooling -> dropout -> dense softmax over 11 classes.

    `weights` may be "imagenet" (downloaded once and cached in ~/.keras),
    a path to a local .h5 weights file (offline Kaggle/Colab), or None.
    """
    try:
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3), include_top=False, weights=weights)
    except Exception as error:
        raise RuntimeError(
            "Could not load MobileNetV2 pretrained weights "
            f"({error.__class__.__name__}: {error}). If this machine has no internet, download "
            "'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5' elsewhere and "
            "pass its path as weights=..., or attach it as a Kaggle dataset.") from None
    backbone._name = BACKBONE_LAYER_NAME
    backbone.trainable = False

    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale_minus1_1")(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="head_pool")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="ovascope_mobilenetv2")


def get_backbone(model: tf.keras.Model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def unfreeze_top_of_backbone(model: tf.keras.Model, from_layer: int) -> int:
    """Make backbone layers [from_layer:] trainable (BatchNorm layers stay frozen).

    Lower layers detect generic edges/colours that transfer well from ImageNet;
    the upper layers encode ImageNet-specific shapes, so those are the ones worth
    adapting to parasite-egg morphology. Returns the number of trainable layers.
    """
    backbone = get_backbone(model)
    backbone.trainable = True
    trainable = 0
    for index, layer in enumerate(backbone.layers):
        is_bn = isinstance(layer, layers.BatchNormalization)
        layer.trainable = index >= from_layer and not is_bn
        trainable += int(layer.trainable)
    return trainable


def count_parameters(model: tf.keras.Model) -> dict:
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    total = sum(int(tf.size(w)) for w in model.weights)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}
