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


def _leer_orientacion(ruta):
    """Orientación EXIF (1..8) de una imagen; 1 si no hay etiqueta o no se puede leer. Solo lee la cabecera.

    Pillow se importa aquí (solo hace falta con APLICAR_EXIF=1) y fuera del try: si falta, falla con un
    ImportError claro en vez de tratar todas las imágenes como sin rotar en silencio.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("APLICAR_EXIF=1 necesita Pillow (pip install pillow).") from e
    try:
        with Image.open(ruta) as im:
            o = int(im.getexif().get(274, 1))
    except Exception:
        return 1
    return o if 1 <= o <= 8 else 1


def _orientaciones(rutas):
    """Vector int32 [n] con la orientación EXIF de cada ruta (todo 1 si APLICAR_EXIF está desactivado)."""
    if not C.APLICAR_EXIF:
        return np.ones(len(rutas), dtype=np.int32)
    return np.asarray([_leer_orientacion(r) for r in rutas], dtype=np.int32)


def _aplicar_orientacion(img, orient):
    """Aplica la orientación EXIF `orient` (1..8) a img [H,W,3], equivalente a PIL.ImageOps.exif_transpose.

    tf.image.rot90 gira en sentido antihorario: EXIF 6 (girar 90° horario) = k=3, EXIF 8 = k=1.
    """
    trans = lambda x: tf.transpose(x, [1, 0, 2])
    ramas = [
        lambda: img,                                         # 0 (no usado) / 1: normal
        lambda: img,                                         # 1
        lambda: tf.image.flip_left_right(img),               # 2: espejo horizontal
        lambda: tf.image.rot90(img, 2),                      # 3: 180°
        lambda: tf.image.flip_up_down(img),                  # 4: espejo vertical
        lambda: trans(img),                                  # 5: transpuesta
        lambda: tf.image.rot90(img, 3),                      # 6: 90° horario
        lambda: tf.image.rot90(trans(img), 2),               # 7: transversa
        lambda: tf.image.rot90(img, 1),                      # 8: 90° antihorario
    ]
    orient = tf.cast(orient, tf.int32)
    orient = tf.where((orient >= 1) & (orient <= 8), orient, 1)   # valor desconocido = sin rotar
    return tf.switch_case(orient, ramas, default=ramas[1])


def _cargar_imagen(ruta, orient=None):
    img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
    if C.APLICAR_EXIF and orient is not None:     # antes del resize; las etiquetas están en el marco orientado
        img = _aplicar_orientacion(img, orient)
        img.set_shape((None, None, 3))
    img = tf.image.resize(img, (C.IMG_SIZE, C.IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0          # [0, 1]; el modelo reescala a [-1, 1]
    img.set_shape((C.IMG_SIZE, C.IMG_SIZE, 3))
    return img


# ------------------------------------------------------------ Aumento de datos
def _random_zoom(img, gt, p=0.5):
    """Zoom/recorte aleatorio para ampliar objetos pequeños (como monedas y bolígrafos)."""
    def _aplicar():
        s = tf.random.uniform((), 0.60, 0.90)  # factor de recorte (zoom de 1.1x a 1.67x)
        valido = gt[:, 0] >= 0
        indices_validos = tf.where(valido)[:, 0]
        n_val = tf.shape(indices_validos)[0]

        def con_objetos():
            # Centra el zoom cerca de un objeto existente
            idx = tf.random.uniform((), 0, n_val, dtype=tf.int32)
            k = tf.gather(indices_validos, idx)
            obj_cx = gt[k, 1]
            obj_cy = gt[k, 2]
            x0 = tf.clip_by_value(obj_cx - s * tf.random.uniform((), 0.2, 0.8), 0.0, 1.0 - s)
            y0 = tf.clip_by_value(obj_cy - s * tf.random.uniform((), 0.2, 0.8), 0.0, 1.0 - s)
            return x0, y0

        def sin_objetos():
            x0 = tf.random.uniform((), 0.0, 1.0 - s)
            y0 = tf.random.uniform((), 0.0, 1.0 - s)
            return x0, y0

        x0, y0 = tf.cond(n_val > 0, con_objetos, sin_objetos)
        x1, y1 = x0 + s, y0 + s

        img_crop = tf.image.crop_and_resize(
            img[None, ...], [[y0, x0, y1, x1]], [0], (C.IMG_SIZE, C.IMG_SIZE)
        )[0]

        # Ajuste de cajas en coordenadas normalizadas del nuevo recorte
        cx_new = (gt[:, 1] - x0) / s
        cy_new = (gt[:, 2] - y0) / s
        w_new = gt[:, 3] / s
        h_new = gt[:, 4] / s

        xmin = tf.maximum(cx_new - w_new / 2.0, 0.0)
        xmax = tf.minimum(cx_new + w_new / 2.0, 1.0)
        ymin = tf.maximum(cy_new - h_new / 2.0, 0.0)
        ymax = tf.minimum(cy_new + h_new / 2.0, 1.0)

        w_clip = tf.maximum(xmax - xmin, 0.0)
        h_clip = tf.maximum(ymax - ymin, 0.0)
        cx_clip = (xmin + xmax) / 2.0
        cy_clip = (ymin + ymax) / 2.0

        # Mantiene la caja si su centro original estaba en el recorte y conserva tamaño visible
        queda_dentro = (gt[:, 1] >= x0) & (gt[:, 1] <= x1) & (gt[:, 2] >= y0) & (gt[:, 2] <= y1)
        sigue_valido = valido & queda_dentro & (w_clip >= 0.02) & (h_clip >= 0.02)

        cls = tf.where(sigue_valido, gt[:, 0], -1.0)
        cx = tf.where(sigue_valido, cx_clip, -1.0)
        cy = tf.where(sigue_valido, cy_clip, -1.0)
        w = tf.where(sigue_valido, w_clip, -1.0)
        h = tf.where(sigue_valido, h_clip, -1.0)

        return img_crop, tf.stack([cls, cx, cy, w, h], axis=1)

    hacer_zoom = tf.random.uniform(()) < p
    img_out, gt_out = tf.cond(hacer_zoom, _aplicar, lambda: (img, gt))
    img_out.set_shape((C.IMG_SIZE, C.IMG_SIZE, 3))
    gt_out.set_shape((C.MAX_CAJAS, 5))
    return img_out, gt_out


def _aumentar(img, gt):
    """Flip horizontal, zoom aleatorio, brillo y contraste aleatorios (solo entrenamiento)."""
    flip = tf.random.uniform(()) < 0.5
    img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
    valido = gt[:, 0] >= 0
    cx = tf.where(valido & flip, 1.0 - gt[:, 1], gt[:, 1])   # refleja el centro x
    gt = tf.concat([gt[:, :1], cx[:, None], gt[:, 2:]], axis=1)
    img, gt = _random_zoom(img, gt, p=0.5)
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
    ds = tf.data.Dataset.from_tensor_slices((rutas, gts, _orientaciones(rutas)))
    if entrenamiento:
        ds = ds.shuffle(len(rutas), reshuffle_each_iteration=True)
    ds = ds.map(lambda r, g, o: (_cargar_imagen(r, o), g), tf.data.AUTOTUNE)
    if entrenamiento:
        ds = ds.map(_aumentar, tf.data.AUTOTUNE)
    ds = ds.map(_codificar_objetivos, tf.data.AUTOTUNE)
    return ds.batch(C.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def dataset_solo_imagenes(rutas):
    """Dataset ordenado (sin shuffle) solo con imágenes, para evaluar."""
    ds = tf.data.Dataset.from_tensor_slices((list(rutas), _orientaciones(rutas)))
    ds = ds.map(_cargar_imagen, tf.data.AUTOTUNE)
    return ds.batch(C.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
