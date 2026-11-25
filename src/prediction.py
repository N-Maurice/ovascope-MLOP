"""Load a saved OvaScope model and classify single images.

    from src.prediction import Predictor
    predictor = Predictor.from_active(cfg)          # the currently activated version
    result = predictor.predict_bytes(open("egg.jpg", "rb").read())

The same Predictor is used by the notebook, the API and the tests, so all three
apply identical preprocessing (src.preprocessing.preprocess_pil_image).

The input must be a crop centred on ONE egg. The model does not detect or count
eggs in a full microscope photo.

Input check: if an egg gate has been trained (models/gate/, see src/gate.py), every image is
checked first. Images that do not look like a parasite-egg crop (a car, a person, an empty
microscope field ...) raise NotParasiteEggError instead of being forced into one of the 11 classes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import registry
from src.gate import EggGate, NotParasiteEggError
from src.preprocessing import load_class_map, load_image_from_bytes, preprocess_pil_image

DISCLAIMER = ("Confidence is the model's softmax score, not a probability of being correct. "
              "OvaScope is a student research prototype, not a validated diagnostic tool.")


class Predictor:
    """Holds one loaded model plus its class mapping and metadata."""

    def __init__(self, version_dir: Path, gate: EggGate | None = None):
        import tensorflow as tf
        version_dir = Path(version_dir)
        model_path = version_dir / "model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}. Train one with `python -m src.train`.")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        # Compile the forward pass once. Calling a Keras model eagerly dispatches its ~150 layers
        # one by one from Python (measured: ~340 ms/image on CPU for MobileNetV2 vs ~16 ms compiled).
        size = int(self.model.input_shape[1])
        self._forward = tf.function(lambda batch: self.model(batch, training=False),
                                    input_signature=[tf.TensorSpec([None, size, size, 3], tf.float32)])
        self.class_map = load_class_map(version_dir / "class_mapping.json")
        self.metadata = json.loads((version_dir / "model_metadata.json").read_text())
        self.version = self.metadata["version"]
        self.image_size = int(self.metadata["input_shape"][0])
        self.gate = gate
        output_classes = self.model.output_shape[-1]
        if output_classes != len(self.class_map):
            raise ValueError(f"Model outputs {output_classes} classes but class mapping has {len(self.class_map)}")

    @staticmethod
    def load_gate(cfg: dict) -> EggGate | None:
        """The trained egg gate, unless disabled (egg_gate.enabled: false) or not trained yet."""
        if not cfg.get("egg_gate", {}).get("enabled", True):
            return None
        return EggGate.from_models_dir(cfg["paths"]["models_dir"])

    @classmethod
    def from_active(cls, cfg: dict) -> "Predictor":
        models_dir = cfg["paths"]["models_dir"]
        version = registry.get_active_version(models_dir)
        if version is None:
            raise FileNotFoundError(f"No active model in {models_dir}. Train and activate a model first.")
        return cls(registry.version_path(models_dir, version), gate=cls.load_gate(cfg))

    @classmethod
    def from_version(cls, cfg: dict, version: str) -> "Predictor":
        return cls(registry.version_path(cfg["paths"]["models_dir"], version), gate=cls.load_gate(cfg))

    def check_input(self, image) -> dict:
        """Run the egg gate. Returns its verdict, or {"enabled": False} when no gate is loaded."""
        if self.gate is None:
            return {"enabled": False}
        return {"enabled": True, **self.gate.check(image)}

    def predict_probabilities(self, image) -> np.ndarray:
        batch = preprocess_pil_image(image, self.image_size)
        return self._forward(batch).numpy()[0]

    def predict_image(self, image, top_k: int = 3, check_input: bool = True) -> dict:
        """Classify one egg crop. With check_input=True (default) the egg gate runs first and
        raises NotParasiteEggError for images that are not parasite-egg crops."""
        input_check = self.check_input(image) if check_input else {"enabled": False}
        if input_check.get("enabled") and not input_check["is_parasite_egg"]:
            raise NotParasiteEggError(input_check)
        probabilities = self.predict_probabilities(image)
        order = np.argsort(probabilities)[::-1][:top_k]
        best = int(order[0])
        entry = self.class_map[best]
        return {
            "predicted_class_id": best,
            "predicted_class": entry["display_name"],
            "class_key": entry["folder"][3:],          # e.g. "ascaris_lumbricoides"
            "original_label": entry["original_name"],
            "confidence": float(probabilities[best]),
            "top_k": [{"class_id": int(i), "class_name": self.class_map[i]["display_name"],
                       "confidence": float(probabilities[i])} for i in order],
            "model_version": self.version,
            "input_check": input_check,
            "note": DISCLAIMER,
        }

    def predict_bytes(self, data: bytes, top_k: int = 3, check_input: bool = True) -> dict:
        """Validate + decode raw file bytes, then predict. Raises ValueError for unreadable input
        and NotParasiteEggError when the egg gate rejects the image."""
        return self.predict_image(load_image_from_bytes(data), top_k=top_k, check_input=check_input)

    def predict_file(self, path, top_k: int = 3, check_input: bool = True) -> dict:
        return self.predict_bytes(Path(path).read_bytes(), top_k=top_k, check_input=check_input)
