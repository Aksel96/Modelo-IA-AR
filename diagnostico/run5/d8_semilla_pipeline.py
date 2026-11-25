"""Diagnóstico 8 (SIN ENTRENAR): ¿qué partes del pipeline de entrenamiento son reproducibles con una semilla?

Sirve para fijar qué se puede garantizar al comparar E6b-EXIF y E6b-Legacy con `train.py --semilla N`.
Prueba por separado: (a) tf.data.shuffle puro, (b) el barajado del pipeline real (`construir_dataset`), (c) los aumentos
aleatorios (flip, zoom, brillo, contraste) en un `map` paralelo con AUTOTUNE.
Uso:  python diagnostico/run5/d8_semilla_pipeline.py
"""
import os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import tensorflow as tf
import config as C, data as D

n = len(D.listar_muestras("train")[0])


def orden(seed):
    tf.keras.utils.set_random_seed(seed)
    ds = tf.data.Dataset.range(n).shuffle(n, reshuffle_each_iteration=True)
    return [int(v) for v in ds.take(10)], [int(v) for v in ds.take(10)]


def sin_aug(seed):
    tf.keras.utils.set_random_seed(seed)
    rutas, gts = D.listar_muestras("train")
    ds = tf.data.Dataset.from_tensor_slices((rutas, gts, D._orientaciones(rutas))).shuffle(n, reshuffle_each_iteration=True)
    return [r.numpy() for r, _, _ in ds.take(8)]


def aug(seed):
    tf.keras.utils.set_random_seed(seed)
    rutas, gts = D.listar_muestras("train")
    ds = tf.data.Dataset.from_tensor_slices((rutas[:16], gts[:16], D._orientaciones(rutas[:16])))
    ds = ds.map(lambda r, g, o: (D._cargar_imagen(r, o), g), tf.data.AUTOTUNE).map(D._aumentar, tf.data.AUTOTUNE)
    return [x.numpy() for x, _ in ds]


print("(a) shuffle puro: misma semilla da el mismo orden:", orden(5)[0] == orden(5)[0])
print("(a) shuffle puro: la 2.ª época difiere de la 1.ª (reshuffle):", orden(5)[0] != orden(5)[1])
print("(b) barajado del pipeline: misma semilla igual:", sin_aug(5) == sin_aug(5), "| semilla distinta igual:", sin_aug(5) == sin_aug(6))
a1, a2 = aug(5), aug(5)
print("(c) aumentos aleatorios (map paralelo), misma semilla: imágenes idénticas", sum(np.array_equal(u, v) for u, v in zip(a1, a2)), "de", len(a1))
