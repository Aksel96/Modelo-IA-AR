"""SSD con backbone MobileNetV2 entrenado DESDE CERO (sin pesos de ImageNet)."""
import tensorflow as tf
from tensorflow.keras import layers, Model

import config as C
from anchors import ANCHORS_CXCYWH, decodificar


def _bloque_extra(x, filtros_intermedios, filtros, nombre):
    """Reduce la resolución a la mitad (stride 2) para detectar objetos más grandes."""
    x = layers.Conv2D(filtros_intermedios, 1, padding="same", use_bias=False, name=f"{nombre}_pw")(x)
    x = layers.BatchNormalization(name=f"{nombre}_bn1")(x)
    x = layers.ReLU(6.0)(x)
    x = layers.SeparableConv2D(filtros, 3, strides=2, padding="same", use_bias=False, name=f"{nombre}_dw")(x)
    x = layers.BatchNormalization(name=f"{nombre}_bn2")(x)
    return layers.ReLU(6.0)(x)


def _cabeza(x, canales_por_anchor, nombre):
    """Cabeza convolucional -> [B, H*W*n_anchors, canales_por_anchor]."""
    n = C.ANCHORS_POR_CELDA
    x = layers.SeparableConv2D(n * canales_por_anchor, 3, padding="same", name=nombre)(x)
    return layers.Reshape((-1, canales_por_anchor))(x)


def construir_ssd(num_clases=C.NUM_CLASES_CON_FONDO, img_size=C.IMG_SIZE):
    """Devuelve un modelo Keras: imagen [0,1] -> [B, A, 4 + num_clases].

    Las primeras 4 columnas son offsets de caja; el resto, logits de clase
    (índice 0 = fondo).
    """
    # IMPORTANTE: weights=None -> inicialización aleatoria, NO se descarga ImageNet.
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=None, alpha=1.0)
    if C.BN_MOMENTUM_BACKBONE is not None:       # E6b: solo las BatchNorm del backbone (ver config.py)
        for capa in backbone.layers:
            if isinstance(capa, layers.BatchNormalization):
                capa.momentum = C.BN_MOMENTUM_BACKBONE

    entrada = layers.Input((img_size, img_size, 3), name="imagen")
    # Entrada en [0,1] -> [-1,1], rango que espera MobileNetV2.
    x = layers.Rescaling(2.0, offset=-1.0, name="normalizar")(entrada)
    backbone_in = Model(backbone.input, [backbone.get_layer("block_13_expand_relu").output,
                                         backbone.output], name="mobilenetv2_scratch")
    f1, f2 = backbone_in(x)                      # strides 16 y 32  -> 16x16 y 8x8
    f3 = _bloque_extra(f2, 128, 256, "extra1")   # 4x4
    f4 = _bloque_extra(f3, 128, 256, "extra2")   # 2x2
    f5 = _bloque_extra(f4, 64, 128, "extra3")    # 1x1
    mapas = [f1, f2, f3, f4, f5]
    tamanos = [m.shape[1] for m in mapas]
    assert tamanos == C.FEATURE_MAPS, f"Mapas {tamanos} != {C.FEATURE_MAPS} (usa IMG_SIZE=256)"

    cajas = layers.Concatenate(axis=1, name="cajas")(
        [_cabeza(m, 4, f"reg_{i}") for i, m in enumerate(mapas)])
    clases = layers.Concatenate(axis=1, name="clases")(
        [_cabeza(m, num_clases, f"cls_{i}") for i, m in enumerate(mapas)])
    salida = layers.Concatenate(axis=-1, name="salida")([cajas, clases])
    return Model(entrada, salida, name="ssd_mobilenetv2_scratch")


class DecodificarSSD(layers.Layer):
    """Offsets+logits -> cajas yxyx normalizadas y probabilidades (softmax)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._anchors = tf.constant(ANCHORS_CXCYWH)

    def call(self, x):
        cajas = decodificar(x[..., :4], self._anchors)
        return cajas, tf.nn.softmax(x[..., 4:], axis=-1)


def construir_modelo_inferencia(modelo):
    """Envuelve el modelo entrenado añadiendo decodificación y softmax.

    Salidas: boxes [B,A,4] (ymin,xmin,ymax,xmax normalizado) y scores [B,A,N+1].
    El NMS se aplica fuera (en la app) por compatibilidad con TFLite.
    """
    cajas, scores = DecodificarSSD(name="decodificar")(modelo.output)
    return Model(modelo.input, [cajas, scores], name="ssd_inferencia")
