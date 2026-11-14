"""Pérdida SSD: Smooth L1 (cajas) + entropía cruzada sparse (clases) con hard negative mining."""
import tensorflow as tf

import config as C


def smooth_l1(diff):
    d = tf.abs(diff)
    return tf.where(d < 1.0, 0.5 * d * d, d - 0.5)


def perdida_ssd(y_true, y_pred):
    """y_true [B,A,5] = (clase, dx, dy, dw, dh); y_pred [B,A,4+K] = (offsets, logits)."""
    cls_true = tf.cast(y_true[..., 0], tf.int32)
    box_true = y_true[..., 1:5]
    box_pred = y_pred[..., :4]
    logits = y_pred[..., 4:]

    positivo = cls_true > 0
    pos_f = tf.cast(positivo, tf.float32)
    n_pos = tf.reduce_sum(pos_f, axis=1)                               # [B]

    # Clasificación: SparseCategoricalCrossentropy por anchor, sin reducir.
    ce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none")(cls_true, logits)          # [B,A]

    # Hard negative mining: solo los negativos con mayor pérdida (ratio 3:1).
    ce_neg = tf.where(positivo, 0.0, ce)
    rango = tf.argsort(tf.argsort(ce_neg, axis=1, direction="DESCENDING"), axis=1)
    n_neg = tf.maximum(C.RATIO_NEGATIVOS * n_pos, 10.0)
    neg_mask = tf.cast(tf.cast(rango, tf.float32) < n_neg[:, None], tf.float32) * (1.0 - pos_f)
    perdida_cls = tf.reduce_sum(ce * (pos_f + neg_mask))

    # Regresión: Smooth L1 solo en anchors positivos.
    perdida_box = tf.reduce_sum(tf.reduce_sum(smooth_l1(box_true - box_pred), axis=-1) * pos_f)

    return (perdida_cls + perdida_box) / tf.maximum(tf.reduce_sum(n_pos), 1.0)
