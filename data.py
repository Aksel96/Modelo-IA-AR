"""Carga de datos YOLO -> tf.data.Dataset con aumento y asignación de anchors.

Estructura esperada:
    data/train/images/*.jpg   data/train/labels/*.txt
    data/val/images/*.jpg     data/val/labels/*.txt
Cada línea de un .txt:  clase cx cy w h   (normalizados 0..1).
Una imagen sin .txt (o vacío) se trata como solo fondo.
"""
import numpy as np
import tensorflow as tf

import config as C
from anchors import ANCHORS_CXCYWH, codificar, cxcywh_a_yxyx, matriz_iou

_EXT = (".jpg", ".jpeg", ".png", ".bmp")
_ANCHORS = tf.constant(ANCHORS_CXCYWH)
_ANCHORS_YXYX = cxcywh_a_yxyx(_ANCHORS)


# ------------------------------------------------------------ Lectura de disco
def _leer_etiquetas(ruta_txt):
    """Devuelve array [n, 5] = (clase, cx, cy, w, h)."""
    filas = []
    if ruta_txt.exists():
        for linea in ruta_txt.read_text().splitlines():
            p = linea.split()
            if len(p) != 5:
                continue
            cls = int(float(p[0]))
            if not 0 <= cls < C.NUM_CLASES:
                raise ValueError(f"Clase {cls} fuera de rango en {ruta_txt}")
            filas.append([cls] + [float(v) for v in p[1:]])
    return np.asarray(filas, dtype=np.float32).reshape(-1, 5)


def listar_muestras(split):
    """Devuelve (rutas_imagen[str], gts[n_img, MAX_CAJAS, 5]) con relleno -1."""
    carpeta = C.DATA_DIR / split
    imgs = sorted(p for p in (carpeta / "images").iterdir() if p.suffix.lower() in _EXT)
    if not imgs:
        raise FileNotFoundError(f"No hay imágenes en {carpeta / 'images'}")
    gts = np.full((len(imgs), C.MAX_CAJAS, 5), -1.0, dtype=np.float32)
    for i, p in enumerate(imgs):
        et = _leer_etiquetas(carpeta / "labels" / (p.stem + ".txt"))[: C.MAX_CAJAS]
        gts[i, : len(et)] = et
    return [str(p) for p in imgs], gts


def _cargar_imagen(ruta):
    img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
    img = tf.image.resize(img, (C.IMG_SIZE, C.IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0          # [0, 1]; el modelo reescala a [-1, 1]
    img.set_shape((C.IMG_SIZE, C.IMG_SIZE, 3))
    return img


# ------------------------------------------------------------ Aumento de datos
def _aumentar(img, gt):
    """Flip horizontal, brillo y contraste aleatorios (solo entrenamiento)."""
    flip = tf.random.uniform(()) < 0.5
    img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
    valido = gt[:, 0] >= 0
    cx = tf.where(valido & flip, 1.0 - gt[:, 1], gt[:, 1])   # refleja el centro x
    gt = tf.concat([gt[:, :1], cx[:, None], gt[:, 2:]], axis=1)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return tf.clip_by_value(img, 0.0, 1.0), gt


# ------------------------------------------------------------ Asignación de anchors
def _codificar_objetivos(img, gt):
    """Asigna cada anchor a un GT y devuelve y [A, 5] = (clase, dx, dy, dw, dh).

    clase 0 = fondo; 1..N = objeto. Un anchor es positivo si IoU >= umbral con
    algún GT; además cada GT fuerza a su mejor anchor a ser positivo.
    """
    valido = gt[:, 0] >= 0
    iou = matriz_iou(_ANCHORS_YXYX, cxcywh_a_yxyx(gt[:, 1:5]))       # [A, M]
    iou = tf.where(valido[None, :], iou, -1.0)
    mejor_gt = tf.argmax(iou, axis=1, output_type=tf.int32)          # [A]
    mejor_iou = tf.reduce_max(iou, axis=1)

    # Cada GT válido fuerza a su mejor anchor.
    mejor_anchor = tf.boolean_mask(tf.argmax(iou, axis=0, output_type=tf.int32), valido)
    ids_gt = tf.boolean_mask(tf.range(C.MAX_CAJAS, dtype=tf.int32), valido)
    idx = mejor_anchor[:, None]
    mejor_iou = tf.tensor_scatter_nd_update(mejor_iou, idx, tf.fill(tf.shape(ids_gt), 2.0))
    mejor_gt = tf.tensor_scatter_nd_update(mejor_gt, idx, ids_gt)

    asignado = tf.gather(gt, mejor_gt)                               # [A, 5]
    positivo = mejor_iou >= C.UMBRAL_IOU_POSITIVO
    cls = tf.where(positivo, asignado[:, 0] + 1.0, 0.0)
    cajas = codificar(asignado[:, 1:5], _ANCHORS)
    cajas = tf.where(positivo[:, None], cajas, 0.0)
    return img, tf.concat([cls[:, None], cajas], axis=1)


# ------------------------------------------------------------ API pública
def construir_dataset(split, entrenamiento):
    rutas, gts = listar_muestras(split)
    ds = tf.data.Dataset.from_tensor_slices((rutas, gts))
    if entrenamiento:
        ds = ds.shuffle(len(rutas), reshuffle_each_iteration=True)
    ds = ds.map(lambda r, g: (_cargar_imagen(r), g), tf.data.AUTOTUNE)
    if entrenamiento:
        ds = ds.map(_aumentar, tf.data.AUTOTUNE)
    ds = ds.map(_codificar_objetivos, tf.data.AUTOTUNE)
    return ds.batch(C.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def dataset_solo_imagenes(rutas):
    """Dataset ordenado (sin shuffle) solo con imágenes, para evaluar."""
    ds = tf.data.Dataset.from_tensor_slices(rutas).map(_cargar_imagen, tf.data.AUTOTUNE)
    return ds.batch(C.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
