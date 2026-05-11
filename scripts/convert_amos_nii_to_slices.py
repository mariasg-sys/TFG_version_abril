# -*- coding: utf-8 -*-

"""
Convierte AMOS en formato NIfTI (.nii.gz) a slices 2D PNG para entrenar una U-Net 2D.

Usa la carpeta real de datos, no __MACOSX:
    D:/Clase/TFG/versión abril/amos/amos22

Salida por defecto:
    D:/Clase/TFG/versión abril/amos_slices_2d

Requiere:
    pip install nibabel pillow numpy
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


AMOS_ROOT = Path("D:/Clase/TFG/versión abril/amos/amos22")
OUTPUT_ROOT = Path("D:/Clase/TFG/versión abril/amos_slices_2d")

WINDOW_MIN = -200
WINDOW_MAX = 300
SLICE_AXIS = 2

# Guardar todos los slices funciona, pero mete muchisimo fondo.
# Si quieres entrenar mas rapido, puedes bajar EMPTY_SLICE_KEEP_EVERY a 5 o 10.
EMPTY_SLICE_KEEP_EVERY = 5


def normalize_ct_to_uint8(volume):
    volume = np.clip(volume, WINDOW_MIN, WINDOW_MAX)
    volume = (volume - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    volume = (volume * 255.0).astype(np.uint8)
    return volume


def get_slice(volume, index, axis):
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    if axis == 2:
        return volume[:, :, index]
    raise ValueError("SLICE_AXIS debe ser 0, 1 o 2")


def save_png(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def convert_split_with_labels(split_name):
    image_dir = AMOS_ROOT / f"images{split_name}"
    label_dir = AMOS_ROOT / f"labels{split_name}"
    output_image_dir = OUTPUT_ROOT / f"images{split_name}"
    output_label_dir = OUTPUT_ROOT / f"labels{split_name}"

    image_paths = sorted(image_dir.glob("amos_*.nii.gz"))
    if not image_paths:
        raise RuntimeError(f"No hay volumenes en {image_dir}")

    for image_path in image_paths:
        label_path = label_dir / image_path.name
        if not label_path.exists():
            print(f"Sin label, salto: {image_path.name}")
            continue

        image_volume = nib.load(str(image_path)).get_fdata(dtype=np.float32)
        label_volume = np.asarray(nib.load(str(label_path)).dataobj, dtype=np.uint8)
        image_volume = normalize_ct_to_uint8(image_volume)

        num_slices = image_volume.shape[SLICE_AXIS]
        saved = 0

        for z in range(num_slices):
            image_slice = get_slice(image_volume, z, SLICE_AXIS)
            label_slice = get_slice(label_volume, z, SLICE_AXIS)

            has_organ = np.any(label_slice > 0)
            if not has_organ and z % EMPTY_SLICE_KEEP_EVERY != 0:
                continue

            stem = image_path.name.replace(".nii.gz", "")
            filename = f"{stem}_z{z:04d}.png"

            save_png(image_slice, output_image_dir / filename)
            save_png(label_slice, output_label_dir / filename)
            saved += 1

        print(f"{split_name} {image_path.name}: {saved}/{num_slices} slices guardados")


def convert_test_images():
    image_dir = AMOS_ROOT / "imagesTs"
    output_image_dir = OUTPUT_ROOT / "imagesTs"

    image_paths = sorted(image_dir.glob("amos_*.nii.gz"))
    if not image_paths:
        print(f"No hay test en {image_dir}")
        return

    for image_path in image_paths:
        image_volume = nib.load(str(image_path)).get_fdata(dtype=np.float32)
        image_volume = normalize_ct_to_uint8(image_volume)

        num_slices = image_volume.shape[SLICE_AXIS]
        for z in range(num_slices):
            image_slice = get_slice(image_volume, z, SLICE_AXIS)
            stem = image_path.name.replace(".nii.gz", "")
            filename = f"{stem}_z{z:04d}.png"
            save_png(image_slice, output_image_dir / filename)

        print(f"Ts {image_path.name}: {num_slices} slices guardados")


def main():
    print(f"Leyendo AMOS desde: {AMOS_ROOT}")
    print(f"Guardando slices en: {OUTPUT_ROOT}")
    convert_split_with_labels("Tr")
    convert_split_with_labels("Va")
    convert_test_images()
    print("Conversion terminada.")


if __name__ == "__main__":
    main()
