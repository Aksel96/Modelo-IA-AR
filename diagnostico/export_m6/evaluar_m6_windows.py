"""Solo lectura: evalúa archivo/best_m6.keras en el .venv de Windows con evaluate.evaluar SIN modificarlo.
Como Keras 3.12 no deserializa la config guardada por Keras 3.16 (WSL), se sustituye load_model (solo en este proceso)
por: reconstruir la arquitectura con construir_ssd() + load_weights del .keras. Requiere APLICAR_EXIF=1."""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["BN_MOMENTUM_BACKBONE"] = "0.99"
sys.path.insert(0, os.getcwd())
import tensorflow as tf
import config as C
assert C.APLICAR_EXIF, "Ejecutar con APLICAR_EXIF=1 (marco de entrenamiento del Run 6)"
from model import construir_ssd
import evaluate


def cargar_por_pesos(ruta, compile=False):
    m = construir_ssd()
    m.load_weights(ruta)
    return m


tf.keras.models.load_model = cargar_por_pesos
evaluate.evaluar("archivo/best_m6.keras", umbral_score=0.5)
