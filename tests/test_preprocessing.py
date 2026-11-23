import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (load_class_map, load_image_from_bytes, make_dataset, preprocess_pil_image,
                               read_manifest, resize_image, square_crop_box, verify_crop_dataset)


@pytest.mark.parametrize("box,size", [([10, 20, 50, 80], (300, 200)), ([0, 0, 30, 30], (100, 100)),
                                      ([280, 180, 40, 40], (300, 200)), ([-5, 10, 400, 50], (300, 200))])
def test_square_crop_box_is_square_and_inside(box, size):
    left, top, right, bottom = square_crop_box(box, 0.1, *size)
    assert right - left == bottom - top
    assert 0 <= left and 0 <= top and right <= size[0] and bottom <= size[1]


def test_crop_dataset_built_and_grouped(project):
    processed = project["cfg"]["paths"]["processed_dir"]
    report = verify_crop_dataset(processed)
    assert report["ok"], report
    rows = read_manifest(processed)
    splits_per_source = {}
    for row in rows:
        if row["split"] != "test":
            splits_per_source.setdefault(row["source_image"], set()).add(row["split"])
    assert all(len(s) == 1 for s in splits_per_source.values())   # no photo in both train and val
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_class_map_preserves_official_names(project):
    class_map = load_class_map(project["cfg"]["paths"]["processed_dir"])
    assert [c["class_id"] for c in class_map] == list(range(11))
    entry = class_map[7]
    assert entry["original_name"] == "Opisthorchis viverrine"
    assert entry["display_name"] == "Opisthorchis viverrini"
    assert entry["folder"] == "07_opisthorchis_viverrini"


def test_resize_output_shape_range_channels():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (97, 131, 3), dtype=np.uint8))
    batch = preprocess_pil_image(image, 224)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert batch.min() >= 0 and batch.max() <= 255


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_non_rgb_images_become_three_channels(mode):
    buffer = io.BytesIO()
    Image.new(mode, (50, 40)).save(buffer, "PNG")
    image = load_image_from_bytes(buffer.getvalue())
    assert image.mode == "RGB"
    assert preprocess_pil_image(image, 64).shape == (1, 64, 64, 3)


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\xff\xd8\xff\xe0" + b"\x00" * 20])
def test_invalid_images_raise_value_error(data):
    with pytest.raises(ValueError):
        load_image_from_bytes(data)


def test_training_and_serving_preprocessing_match(project):
    """The tf.data path (training/evaluation) and the PIL path (API) must give the same pixels."""
    processed = project["cfg"]["paths"]["processed_dir"]
    row = read_manifest(processed)[0]
    path = str(processed / row["crop_path"])
    from_dataset = next(iter(make_dataset([path], [0], 224, 1, training=False)))[0].numpy()[0]
    with open(path, "rb") as f:
        from_serving = preprocess_pil_image(load_image_from_bytes(f.read()), 224)[0]
    assert np.abs(from_dataset - from_serving).max() < 1e-3   # identical decoders + identical resize


def test_augmentation_only_in_training(project):
    processed = project["cfg"]["paths"]["processed_dir"]
    path = str(processed / read_manifest(processed)[0]["crop_path"])
    plain_a = next(iter(make_dataset([path], [0], 64, 1, training=False)))[0].numpy()
    plain_b = next(iter(make_dataset([path], [0], 64, 1, training=False)))[0].numpy()
    assert np.array_equal(plain_a, plain_b)
    augmented = next(iter(make_dataset([path], [0], 64, 1, training=True)))[0].numpy()
    assert augmented.shape == plain_a.shape and augmented.min() >= 0 and augmented.max() <= 255


def test_resize_is_deterministic():
    array = np.random.default_rng(1).integers(0, 256, (50, 50, 3), dtype=np.uint8)
    assert np.array_equal(resize_image(array, 32).numpy(), resize_image(array, 32).numpy())
