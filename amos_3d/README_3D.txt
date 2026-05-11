Version 3D para AMOS
====================

Esta carpeta contiene una version inicial 3D basada en la version 2D:

1. prepare_amos_3d_patches.py
   Convierte los volumenes .nii.gz de AMOS a parches 3D .npz.

2. Unet3D.py
   U-Net 3D multiclase.

3. utils_3d.py
   Dataset 3D, Dice + CrossEntropy, metricas, CSV y checkpoints top 2.

4. train_3d.py
   Entrenamiento 3D por parches.

Orden de ejecucion:

    cd /d "D:\Clase\TFG\versión abril\amos_3d"
    python prepare_amos_3d_patches.py
    python train_3d.py

Diferencia importante con 2D:

La version 2D entrena con slices PNG [B, 1, H, W].
La version 3D entrena con parches volumetricos [B, 1, D, H, W].

Por memoria GPU, no se entrenan volumenes completos, sino parches 3D.
Si da error CUDA out of memory, baja PATCH_SIZE en prepare_amos_3d_patches.py
o usa features=(8, 16, 32, 64) en train_3d.py.
