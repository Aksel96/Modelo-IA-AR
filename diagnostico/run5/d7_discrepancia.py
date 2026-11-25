"""Diagnóstico 7 (SIN ENTRENAR: solo inferencia): ¿por qué `evaluate.py` directo (mAP 0.055 / 0.056) y el "Original" de d6
(0.062 / 0.065) difieren para el MISMO checkpoint?

Datos previos (score 0.5, IoU 0.5 en todos; el score solo afecta a precisión/recall, no al AP: evaluate.py usa todas las
detecciones con score > 0.01):
    Run 5  directo  : boli 0.001 laptop 0.163 libro 0.109 moneda 0.000 mouse 0.003 -> 0.055
    Run 5  d6 orig. : boli 0.001 laptop 0.189 libro 0.117 moneda 0.000 mouse 0.003 -> 0.062
    Run 4  directo  : boli 0.001 laptop 0.179 libro 0.099 moneda 0.000 mouse 0.003 -> 0.056
    Run 4  d6 orig. : boli 0.001 laptop 0.207 libro 0.114 moneda 0.000 mouse 0.003 -> 0.065
En ambos runs d6 da más en laptop y libro (mismo sentido): puede no ser ruido aleatorio. La diferencia entre las dos rutas es
que d6 hace cargar -> modelo.save(tmp) -> evaluar(tmp), y lo hace tras otras pasadas con el modelo en el mismo proceso.

Este script, en UN solo proceso, evalúa el mismo checkpoint de varias formas y compara las salidas crudas:
  V1, V2  evaluar(ruta original) dos veces seguidas          -> ruido de repetición
  V3, V4  cargar -> guardar en --tmp -> evaluar(copia) x2    -> efecto de re-guardar
  V5      tras pasadas hacia adelante de otro modelo con training=True (como d6), evaluar(ruta original)
  Comparación de pesos (original vs copia) y de las salidas crudas de predict (cajas y scores) entre variantes.
Uso (misma bandera EXIF con la que se entrenó):
    APLICAR_EXIF=1 python diagnostico/run5/d7_discrepancia.py --modelo archivo/best_m5.keras --etiqueta run5
"""
import argparse, contextlib, io, os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
import numpy as np
import tensorflow as tf
import config as C, data as D, evaluate as E
from model import construir_modelo_inferencia, construir_ssd

SALIDA = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); SALIDA.append(s)


def evaluar(ruta):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, _, mapa = E.evaluar(str(ruta), umbral_score=0.5)
    aps = {l.split()[1]: float(l.split(":")[1].split()[0]) for l in buf.getvalue().splitlines() if l.strip().startswith("AP@")}
    return mapa, aps


def crudo(ruta, rutas):
    inf = construir_modelo_inferencia(tf.keras.models.load_model(str(ruta), compile=False))
    cajas, scores = inf.predict(D.dataset_solo_imagenes(rutas), verbose=0)
    return cajas, scores


def fila(nombre, mapa, aps):
    P(f"{nombre:<52}" + "".join(f"{aps[c]:>10.3f}" for c in C.CLASES) + f"{mapa:>10.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="aleatorio"); ap.add_argument("--etiqueta", default="modelo")
    ap.add_argument("--tmp", default="/tmp"); ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--salida", default=str(RAIZ / "diagnostico" / "run5"))
    a = ap.parse_args()
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    if a.rapido:                                   # solo para probar el script: 32 imágenes de val
        _l = E.listar_muestras; E.listar_muestras = lambda s: tuple(x[:32] for x in _l(s))
    if a.modelo == "aleatorio":
        orig = tmp / "_d7_aleatorio.keras"; construir_ssd().save(str(orig))
    else:
        orig = Path(a.modelo)
    copia = tmp / "_d7_copia.keras"
    P(f"Etiqueta {a.etiqueta} | modelo {orig} | APLICAR_EXIF={C.APLICAR_EXIF} | TF {tf.__version__} | "
      f"GPU: {[d.name for d in tf.config.list_physical_devices('GPU')]}")
    P(f"{'variante (score 0.5, IoU 0.5)':<52}" + "".join(f"{c:>10}" for c in C.CLASES) + f"{'mAP':>10}")

    fila("V1 evaluar(original)", *evaluar(orig))
    fila("V2 evaluar(original) otra vez", *evaluar(orig))
    m = tf.keras.models.load_model(str(orig), compile=False); m.save(str(copia))
    fila("V3 evaluar(copia re-guardada)", *evaluar(copia))
    fila("V4 evaluar(copia re-guardada) otra vez", *evaluar(copia))
    otro = tf.keras.models.load_model(str(orig), compile=False)          # actividad previa, como en d6 (sección A)
    for k, (x, _y) in enumerate(D.construir_dataset("train", False)):
        otro(x, training=True)
        if k >= 5: break
    fila("V5 evaluar(original) tras actividad training=True", *evaluar(orig))

    P("\n--- Comparación de pesos: original vs copia re-guardada ---")
    a_, b_ = tf.keras.models.load_model(str(orig), compile=False), tf.keras.models.load_model(str(copia), compile=False)
    difs = [float(np.max(np.abs(x.numpy() - y.numpy()))) for x, y in zip(a_.weights, b_.weights)]
    P(f"pesos: {len(a_.weights)} vs {len(b_.weights)} | diferencia máxima absoluta entre pesos: {max(difs):.3e} | "
      f"pesos distintos: {sum(d > 0 for d in difs)}")
    P(f"momentum BN original vs copia iguales: {[l.momentum for l in a_.layers if isinstance(l, tf.keras.layers.BatchNormalization)] == [l.momentum for l in b_.layers if isinstance(l, tf.keras.layers.BatchNormalization)]}")

    P("\n--- Salidas crudas de predict (cajas, scores) sobre val ---")
    rutas, _ = E.listar_muestras("val")
    c1, s1 = crudo(orig, rutas); c2, s2 = crudo(orig, rutas); c3, s3 = crudo(copia, rutas)
    for nombre, (ca, sa), (cb, sb) in (("original vs original (repetición)", (c1, s1), (c2, s2)),
                                       ("original vs copia re-guardada", (c1, s1), (c3, s3))):
        P(f"{nombre:<36} max|dif cajas| {np.max(np.abs(ca - cb)):.3e} | max|dif scores| {np.max(np.abs(sa - sb)):.3e} | "
          f"scores>0.01 (clases 1..N): {int((sa[:, :, 1:] > 0.01).sum())} vs {int((sb[:, :, 1:] > 0.01).sum())}")
    P(f"score máximo de clase en val (original): {float(s1[:, :, 1:].max()):.4f} | mediana del máximo por imagen: "
      f"{float(np.median(s1[:, :, 1:].max(axis=(1, 2)))):.4f}")
    out = Path(a.salida) / f"d7_{a.etiqueta}.txt"; Path(a.salida).mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(SALIDA), encoding="utf-8"); print(f"\nEscrito: {out}")
