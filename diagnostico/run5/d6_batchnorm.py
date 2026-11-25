"""Diagnóstico 6 (SIN ENTRENAR: solo pasadas hacia adelante, sin gradientes ni cambio de pesos): ¿explica BatchNorm
la diferencia entre la `loss` de entrenamiento del historial y la pérdida del mismo checkpoint en modo inferencia?

Contexto: en el Run 5 el historial da loss=4.87 en la época 8, pero d3 mide 7.99 sobre el mismo train sin aumentos
con el checkpoint de la época 8. Los BN del backbone MobileNetV2 usan momentum=0.999 (keras.applications); con 69
pasos por época, tras 8 épocas (552 pasos) las estadísticas móviles conservan 0.999**552 = 58 % de su valor inicial.

Qué hace (todo en memoria; no guarda nada en archivo/ ni en salida/; los temporales van a --tmp):
  A. Pérdida de train y val con 4 combinaciones: (sin/con aumentos) x (BN en modo inferencia / BN con estadísticas del lote).
  B. Recalibra las estadísticas móviles de BN (momentum 0.9, 3 pasadas sobre train sin aumentos, forward con training=True;
     los pesos entrenables NO cambian) y repite pérdida y evaluate.py (mAP, precisión, recall) antes y después.
Uso (misma bandera EXIF con la que se entrenó):
    APLICAR_EXIF=1 python diagnostico/run5/d6_batchnorm.py --modelo archivo/best_m5.keras --etiqueta run5
    python diagnostico/run5/d6_batchnorm.py --modelo /tmp/run4.keras --etiqueta run4
"""
import argparse, contextlib, io, os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
import numpy as np
import tensorflow as tf
import config as C, data as D, evaluate as E
from losses import perdida_ssd

SALIDA = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); SALIDA.append(s)


def bn_layers(m):
    res = []
    for l in m.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            res.append(l)
        elif hasattr(l, "layers"):
            res += bn_layers(l)
    return res


def lotes(split, aug, semilla=0):
    tf.random.set_seed(semilla)
    if aug:
        return D.construir_dataset(split, entrenamiento=True)
    return D.construir_dataset(split, False).unbatch().shuffle(2000, seed=semilla).batch(C.BATCH_SIZE)


def perdida(modelo, split, aug, training):
    tot = n = 0
    for x, y in lotes(split, aug):
        p = modelo(x, training=training)
        b = int(x.shape[0]); tot += float(perdida_ssd(y, p)) * b; n += b
    return tot / n


def cargar(path):
    if path == "aleatorio":
        from model import construir_ssd
        return construir_ssd()
    return tf.keras.models.load_model(path, compile=False)


def evaluar_en_ruta(modelo, tmp, score):
    ruta = str(Path(tmp) / "_tmp_recal.keras"); modelo.save(ruta)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = E.evaluar(ruta, umbral_score=score)
    Path(ruta).unlink(missing_ok=True)
    return buf.getvalue().strip(), r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="aleatorio"); ap.add_argument("--etiqueta", default="modelo")
    ap.add_argument("--tmp", default="/tmp"); ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--salida", default=str(RAIZ / "diagnostico" / "run5"))
    a = ap.parse_args()
    P(f"Etiqueta {a.etiqueta} | modelo {a.modelo} | APLICAR_EXIF={C.APLICAR_EXIF} | TF {tf.__version__}")
    if a.rapido:                                   # solo para probar el script
        _ds = D.construir_dataset; D.construir_dataset = lambda s, entrenamiento: _ds(s, entrenamiento).take(2)

    m0 = cargar(a.modelo); bns = bn_layers(m0)
    mom = sorted({float(l.momentum) for l in bns})
    P(f"Capas BatchNorm: {len(bns)} | momentum distintos: {mom} | pasos/época (train, batch {C.BATCH_SIZE}): {int(np.ceil(1097/C.BATCH_SIZE))}")
    for ep_ in (8, 11, 38):
        pasos = ep_ * int(np.ceil(1097 / C.BATCH_SIZE))
        P(f"  peso residual del valor inicial de las estadísticas móviles tras {ep_} épocas ({pasos} pasos): "
          f"momentum 0.999 -> {0.999**pasos:.2f} | momentum 0.99 -> {0.99**pasos:.4f}")

    P("\n===== A. Pérdida del checkpoint según modo de BN y aumentos (lotes mezclados) =====")
    P(f"{'conjunto':<9}{'aumentos':<10}{'BN inferencia (medias móviles)':>32}{'BN con estadísticas del lote':>32}")
    for split, aug in (("train", False), ("train", True), ("val", False)):
        li = perdida(cargar(a.modelo), split, aug, training=False)
        lb = perdida(cargar(a.modelo), split, aug, training=True)      # modelo recién cargado cada vez
        P(f"{split:<9}{'sí' if aug else 'no':<10}{li:>32.4f}{lb:>32.4f}")
    P("(Referencia: historial_m5.csv época 8 -> loss 4.8737 con aumentos y BN de lote; val_loss 7.9220 con BN de inferencia.)")

    P("\n===== B. Recalibración de las estadísticas móviles de BN (pesos entrenables intactos) =====")
    antes = {l.name: (l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy()) for l in bns}
    txt0, r0 = evaluar_en_ruta(cargar(a.modelo), a.tmp, 0.5) if not a.rapido else ("(omitido en --rapido)", None)
    P("Original (evaluate.py, score 0.5):\n" + txt0)
    m = cargar(a.modelo); b2 = bn_layers(m)
    for l in b2:
        l.momentum = 0.9
    for _ in range(3):
        for x, _y in lotes("train", False, semilla=_):
            m(x, training=True)
    cambios = []
    for l in b2:
        mm0, vv0 = antes[l.name]; mm1, vv1 = l.moving_mean.numpy(), l.moving_variance.numpy()
        cambios.append((l.name, float(np.mean(np.abs(mm1 - mm0)) / (np.mean(np.abs(mm1)) + 1e-6)), float(np.median(vv1 / np.maximum(vv0, 1e-6)))))
    cambios.sort(key=lambda t: -t[1])
    P(f"Cambio relativo medio de la media móvil por capa BN: mediana {np.median([c[1] for c in cambios]):.2f}, "
      f"máximo {cambios[0][1]:.2f} ({cambios[0][0]}); cociente mediano nueva/antigua de la varianza móvil: "
      f"mediana de capas {np.median([c[2] for c in cambios]):.2f}")
    P(f"{'conjunto':<8}{'pérdida con estadísticas recalibradas (BN inferencia)':>60}")
    for split in ("train", "val"):
        P(f"{split:<8}{perdida(m, split, False, training=False):>60.4f}")
    txt1, r1 = evaluar_en_ruta(m, a.tmp, 0.5) if not a.rapido else ("(omitido en --rapido)", None)
    P("Recalibrado (evaluate.py, score 0.5):\n" + txt1)
    if not a.rapido:
        t2, _ = evaluar_en_ruta(m, a.tmp, 0.2); P("Recalibrado (score 0.2):\n" + "\n".join(t2.splitlines()[-3:]))
    out = Path(a.salida) / f"d6_{a.etiqueta}.txt"; Path(a.salida).mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(SALIDA), encoding="utf-8"); print(f"\nEscrito: {out}")
