"""Solo lectura: verifica salida/detector.tflite (Run 6) en el .venv de Windows.
1) firma (entrada/salidas, identificadas por shape); 2) comparación numérica con el modelo Keras (pesos de
archivo/best_m6.keras sobre la arquitectura de model.py); 3) mAP@0.5 en val usando evaluate.evaluar sin modificarlo,
sustituyendo solo en este proceso la predicción por el intérprete TFLite. Requiere APLICAR_EXIF=1."""
import os, sys, hashlib
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["BN_MOMENTUM_BACKBONE"] = "0.99"
sys.path.insert(0, os.getcwd())
import numpy as np, tensorflow as tf
import config as C
assert C.APLICAR_EXIF, "Ejecutar con APLICAR_EXIF=1"
from data import dataset_solo_imagenes, listar_muestras
from model import construir_ssd, construir_modelo_inferencia
import evaluate

TFL = "salida/detector.tflite"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
print("detector.tflite SHA256:", sha(TFL), "|", os.path.getsize(TFL), "bytes")
print("labels.txt:", open(C.LABELS_PATH, encoding="utf-8").read().split())
print("best_model.keras SHA256:", sha("salida/best_model.keras"))

it = tf.lite.Interpreter(model_path=TFL)
it.allocate_tensors()
ent = it.get_input_details()
sal = it.get_output_details()
print("Entrada:", [(d["name"], list(d["shape"]), d["dtype"].__name__) for d in ent])
print("Salidas:", [(d["name"], list(d["shape"]), d["dtype"].__name__) for d in sal])
i_box = next(d["index"] for d in sal if d["shape"][-1] == 4)
i_sc = next(d["index"] for d in sal if d["shape"][-1] == C.NUM_CLASES_CON_FONDO)


def tflite_pred(x):
    cajas, scores = [], []
    for img in x:
        it.set_tensor(ent[0]["index"], img[None].astype(np.float32))
        it.invoke()
        cajas.append(it.get_tensor(i_box)[0]); scores.append(it.get_tensor(i_sc)[0])
    return np.stack(cajas), np.stack(scores)


rutas, _ = listar_muestras("val")
imgs = np.concatenate([b.numpy() for b in dataset_solo_imagenes(rutas)])
print("Imágenes de val:", imgs.shape)
cj_t, sc_t = tflite_pred(imgs)

m = construir_ssd(); m.load_weights("archivo/best_m6.keras")
cj_k, sc_k = construir_modelo_inferencia(m).predict(imgs, batch_size=16, verbose=0)
print(f"|Δ cajas| máx {np.abs(cj_t - cj_k).max():.4f}  medio {np.abs(cj_t - cj_k).mean():.6f}")
print(f"|Δ scores| máx {np.abs(sc_t - sc_k).max():.4f}  medio {np.abs(sc_t - sc_k).mean():.6f}")
print("Misma clase top-1 por anchor:", f"{(sc_t.argmax(-1) == sc_k.argmax(-1)).mean():.4%}")
print("Suma softmax ≈ 1:", bool(np.allclose(sc_t.sum(-1), 1, atol=1e-3)))


class _InfTFLite:
    def predict(self, ds, verbose=0):
        return cj_t, sc_t


tf.keras.models.load_model = lambda *a, **k: None
evaluate.construir_modelo_inferencia = lambda _m: _InfTFLite()
print("\n=== mAP del .tflite (val, APLICAR_EXIF=1, score 0.5, IoU 0.5) ===")
evaluate.evaluar(TFL, umbral_score=0.5)
