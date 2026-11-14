"""Exporta a TFLite + labels.txt.  Uso: python export_tflite.py [--cuantizar]"""
import argparse
import shutil
import tempfile

import tensorflow as tf

import config as C
from model import construir_modelo_inferencia


def exportar(modelo_path, salida, cuantizar):
    modelo = tf.keras.models.load_model(modelo_path, compile=False)
    inf = construir_modelo_inferencia(modelo)       # incluye decodificación + softmax

    tmp = tempfile.mkdtemp()
    try:
        if hasattr(inf, "export"):                  # Keras 3
            inf.export(tmp)
        else:                                       # tf.keras legacy
            tf.saved_model.save(inf, tmp)
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
        if cuantizar:                               # cuantización dinámica de pesos
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite = conv.convert()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(tflite)
    # labels.txt: línea 0 = fondo, luego las clases (mismo orden que los scores).
    C.LABELS_PATH.write_text("\n".join(["fondo"] + C.CLASES) + "\n")
    print(f"TFLite: {salida} ({len(tflite) / 1e6:.2f} MB) | labels: {C.LABELS_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(C.MEJOR_MODELO))
    ap.add_argument("--salida", default=str(C.TFLITE_PATH))
    ap.add_argument("--cuantizar", action="store_true")
    a = ap.parse_args()
    from pathlib import Path
    exportar(a.modelo, Path(a.salida), a.cuantizar)
