"""Huella (SHA256) de la asignación de anchors de TODO el dataset, sin imágenes ni modelo.
Sirve para comprobar si el resultado es idéntico entre entornos (Windows CPU frente a WSL2). Solo lectura.
Uso (desde la raíz del repo):  python diagnostico/run5/d1b_hash_asignacion.py"""
import hashlib, os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, tensorflow as tf
import config as C, data as D

img = tf.zeros((C.IMG_SIZE, C.IMG_SIZE, 3))
h = hashlib.sha256(); n = 0
for split in ("train", "val"):
    _, gts = D.listar_muestras(split)
    for g in gts:
        _, y = D._codificar_objetivos(img, tf.constant(g))
        h.update(np.round(y.numpy(), 5).astype(np.float32).tobytes()); n += 1
print(f"TF {tf.__version__} | imágenes {n} | SHA256 asignación: {h.hexdigest()}")
