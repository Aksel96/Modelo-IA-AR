"""Verificación de configuración de E6b (SIN ENTRENAR; solo construye el modelo y lo inspecciona).

Imprime las banderas activas, el momentum de las BatchNorm del backbone y de los bloques extra, y una huella SHA256 de los
pesos iniciales con una semilla fija: dos ejecuciones con la misma semilla deben dar la misma huella (también con distinta
bandera EXIF), que es la condición de partida para comparar E6b-EXIF y E6b-Legacy.
Uso:  [APLICAR_EXIF=0|1] [BN_MOMENTUM_BACKBONE=0.99] python diagnostico/e6b_verificar_config.py [--semilla 42]
"""
import argparse, hashlib, os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tensorflow as tf
import config as C
from model import construir_ssd

ap = argparse.ArgumentParser(); ap.add_argument("--semilla", type=int, default=42); a = ap.parse_args()
tf.keras.utils.set_random_seed(a.semilla)
m = construir_ssd()
bn = lambda capa: [c for c in capa.layers if isinstance(c, tf.keras.layers.BatchNormalization)]
back = bn(m.get_layer("mobilenetv2_scratch")); extra = bn(m)
h = hashlib.sha256()
for w in m.get_weights():
    h.update(w.tobytes())
print(f"TF {tf.__version__} | Keras {tf.keras.__version__}")
print(f"APLICAR_EXIF={int(C.APLICAR_EXIF)} | BN_MOMENTUM_BACKBONE={C.BN_MOMENTUM_BACKBONE} (None = 0.999 de Keras) | semilla={a.semilla}")
print(f"BatchNorm del backbone: {len(back)}, momentum {sorted({c.momentum for c in back})}")
print(f"BatchNorm de bloques extra: {len(extra)}, momentum {sorted({c.momentum for c in extra})}")
print(f"Pesos: {len(m.weights)} | huella SHA256 de los pesos iniciales: {h.hexdigest()}")
