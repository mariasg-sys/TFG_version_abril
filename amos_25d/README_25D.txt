Version 2.5D para AMOS
======================

Esta version usa los slices PNG ya generados para la version 2D:

    D:/Clase/TFG/versión abril/amos_slices_2d/imagesTr
    D:/Clase/TFG/versión abril/amos_slices_2d/labelsTr
    D:/Clase/TFG/versión abril/amos_slices_2d/imagesVa
    D:/Clase/TFG/versión abril/amos_slices_2d/labelsVa

No hace falta reconvertir los NIfTI si esa carpeta ya existe.

Idea:

    Entrada = [slice z-1, slice z, slice z+1]
    Mascara = label del slice central z

Es decir, la red sigue siendo 2D, pero recibe contexto entre cortes.

Orden de ejecucion:

    cd /d "D:\Clase\TFG\versión abril\amos_25d"
    python train_25d.py

Los resultados se guardan en:

    D:/Clase/TFG/versión abril/amos_25d/runs

Incluye:

    - checkpoints top 2 por val_loss
    - checkpoints top 2 por mean_dice
    - CSV con metricas por epoch
    - predicciones visuales
