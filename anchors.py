"""Anchors SSD, codificación/decodificación de cajas e IoU (TensorFlow)."""
import math

import numpy as np
import tensorflow as tf

import config as C


def generar_anchors() -> np.ndarray:
    """Devuelve anchors [A, 4] en formato (cx, cy, w, h) normalizado.

    Orden: por mapa -> fila y -> columna x -> k-ésimo anchor. Coincide con el
    reshape de las cabezas del modelo ([H, W, n*C] -> [H*W*n, C]).
    """
    escalas = np.linspace(C.ESCALA_MIN, C.ESCALA_MAX, len(C.FEATURE_MAPS) + 1)
    anchors = []
    for k, fs in enumerate(C.FEATURE_MAPS):
        s, s_sig = escalas[k], escalas[k + 1]
        r2 = math.sqrt(2.0)
        whs = [(s, s), (math.sqrt(s * s_sig),) * 2, (s * r2, s / r2), (s / r2, s * r2)]
        assert len(whs) == C.ANCHORS_POR_CELDA
        for y in range(fs):
            for x in range(fs):
                cx, cy = (x + 0.5) / fs, (y + 0.5) / fs
                for w, h in whs:
                    anchors.append((cx, cy, w, h))
    return np.asarray(anchors, dtype=np.float32)


ANCHORS_CXCYWH = generar_anchors()          # [A, 4]
NUM_ANCHORS = ANCHORS_CXCYWH.shape[0]


def cxcywh_a_yxyx(b):
    """(cx, cy, w, h) -> (ymin, xmin, ymax, xmax); formato de tf.image.nms."""
    cx, cy, w, h = tf.unstack(b, axis=-1)
    return tf.stack([cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2], axis=-1)


def matriz_iou(a, b):
    """IoU entre cajas yxyx a [N,4] y b [M,4] -> [N,M]."""
    a = a[:, None, :]
    b = b[None, :, :]
    ymin = tf.maximum(a[..., 0], b[..., 0])
    xmin = tf.maximum(a[..., 1], b[..., 1])
    ymax = tf.minimum(a[..., 2], b[..., 2])
    xmax = tf.minimum(a[..., 3], b[..., 3])
    inter = tf.maximum(ymax - ymin, 0.0) * tf.maximum(xmax - xmin, 0.0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / tf.maximum(area_a + area_b - inter, 1e-9)


def codificar(gt_cxcywh, anchors_cxcywh):
    """Caja GT -> offsets (dx, dy, dw, dh) relativos al anchor."""
    vc, vs = C.VARIANZAS
    g = gt_cxcywh
    a = anchors_cxcywh
    gw = tf.maximum(g[..., 2], 1e-4)
    gh = tf.maximum(g[..., 3], 1e-4)
    dx = (g[..., 0] - a[..., 0]) / (a[..., 2] * vc)
    dy = (g[..., 1] - a[..., 1]) / (a[..., 3] * vc)
    dw = tf.math.log(gw / a[..., 2]) / vs
    dh = tf.math.log(gh / a[..., 3]) / vs
    return tf.stack([dx, dy, dw, dh], axis=-1)


def decodificar(deltas, anchors_cxcywh):
    """Offsets [..., A, 4] -> cajas yxyx normalizadas [..., A, 4]."""
    vc, vs = C.VARIANZAS
    a = anchors_cxcywh
    cx = deltas[..., 0] * vc * a[..., 2] + a[..., 0]
    cy = deltas[..., 1] * vc * a[..., 3] + a[..., 1]
    w = tf.exp(tf.clip_by_value(deltas[..., 2] * vs, -10.0, 10.0)) * a[..., 2]
    h = tf.exp(tf.clip_by_value(deltas[..., 3] * vs, -10.0, 10.0)) * a[..., 3]
    return cxcywh_a_yxyx(tf.stack([cx, cy, w, h], axis=-1))
