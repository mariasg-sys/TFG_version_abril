# -*- coding: utf-8 -*-

"""
Covierte los NIfTI de AMOS en cortes 2D para segmentación en U-Net 2D.

Usa la carpeta real de datos, no __MACOSX:
    D:/Clase/TFG/versión abril/amos/amos22

Salida por defecto:
    D:/Clase/TFG/versión abril/amos_slices_2d

Requiere tener instalados:
     · nibabel 
     · pillow 
     · numpy
     
Idea principal:
    CT 3D NIfTI -> ventana HU -> normalización uint8 -> corte 2D -> PNG
    Label 3D NIfTI -> corte 2D -> PNG
"""
import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image

# directorios
AMOS_ROOT = Path("D:/Clase/TFG/versión abril/amos/amos22")
OUTPUT_ROOT = Path("D:/Clase/TFG/versión abril/amos_slices_2d")

#definición de ventana de intensidades para CT (resalta órganos blandos y descarta hueso y aire)
WINDOW_MIN = -200
WINDOW_MAX = 300
SLICE_AXIS = 2

# Para entrenar más rápido, guardamos cortes de sólo fondo cada cinco, ya que no son representativos para el entrenamiento.
EMPTY_SLICE_KEEP_EVERY = 5

# Pasa de unidades de Houndsfild a imagen de 8 bits (valores entre 0 y 255, fomato png escala de grises)
def normalize_ct_to_uint8(volume):
    volume = np.clip(volume, WINDOW_MIN, WINDOW_MAX)
    volume = (volume - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    volume = (volume * 255.0).astype(np.uint8)
    return volume

# Cortes -> extrae un corte axial por cada z
# Con esta función se puede cambiar fácilmente el eje de corte, 
# elegimos z ya que es la orientaión más habitual a la hora de interpretarlos, ya que las anotaciones suelen verse bien en este corte
def get_slice(volume, index, axis):
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    if axis == 2:
        return volume[:, :, index]
    raise ValueError("SLICE_AXIS debe ser 0, 1 o 2") # si el eje no es válido pasa un error

# Guardado
def save_png(array, path):
    path.parent.mkdir(parents=True, exist_ok=True) #crea la carpeta si no existe
    Image.fromarray(array).save(path)

# Convierte 3D a 2D y asegura que máscaras y etiquetas están alineadas
def convert_split_with_labels(split_name):
    image_dir = AMOS_ROOT / f"images{split_name}"
    label_dir = AMOS_ROOT / f"labels{split_name}"
    output_image_dir = OUTPUT_ROOT / f"images{split_name}"
    output_label_dir = OUTPUT_ROOT / f"labels{split_name}"

    image_paths = sorted(image_dir.glob("amos_*.nii.gz"))
    if not image_paths:
        raise RuntimeError(f"No hay volumenes en {image_dir}")  #si no existe etiqueta con el mismo nombre lo salta

    for image_path in image_paths:
        label_path = label_dir / image_path.name
        if not label_path.exists():
            print(f"Sin label, salto: {image_path.name}")
            continue

        image_volume = nib.load(str(image_path)).get_fdata(dtype=np.float32)
        label_volume = np.asarray(nib.load(str(label_path)).dataobj, dtype=np.uint8)
        image_volume = normalize_ct_to_uint8(image_volume) #normalización a 8 bits

        num_slices = image_volume.shape[SLICE_AXIS]
        saved = 0

        for z in range(num_slices):
            image_slice = get_slice(image_volume, z, SLICE_AXIS) #extracción de corte de imagen
            label_slice = get_slice(label_volume, z, SLICE_AXIS) #extracción de corte de etiqueta

            has_organ = np.any(label_slice > 0)
            if not has_organ and z % EMPTY_SLICE_KEEP_EVERY != 0: # de los vacíos conserva 1/5 para no llenar el dataset de imagenes inútiles
                continue

            stem = image_path.name.replace(".nii.gz", "")
            filename = f"{stem}_z{z:04d}.png"

            save_png(image_slice, output_image_dir / filename)
            save_png(label_slice, output_label_dir / filename)  #guarda los 2 png con el mismo nombre
            saved += 1

        print(f"{split_name} {image_path.name}: {saved}/{num_slices} cortes guardados")

# Lo mismo pero para test, donde normalmente no hay máscaras disponibles, si el dataset no incluye test, se lo salta sin avisar.
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
    print(f"Guardando cortes en: {OUTPUT_ROOT}")
    convert_split_with_labels("Tr")
    convert_split_with_labels("Va")
    convert_test_images()
    print("Conversion terminada.")


if __name__ == "__main__":
    main()
