"""Compara la asignación de anchors entre entornos SIN el redondeo del hash (solo lectura).
  Generar:   python diagnostico/run5/d1c_asignacion_npz.py --etiqueta windows|wsl
  Comparar:  python diagnostico/run5/d1c_asignacion_npz.py --comparar diagnostico/run5/d1c_windows.npz diagnostico/run5/d1c_wsl.npz
Guarda solo los anchors positivos (índice, clase, offsets) de las 1,410 imágenes (unos KB)."""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--etiqueta"); ap.add_argument("--comparar", nargs=2)
a = ap.parse_args()
if a.comparar:
    A, B = np.load(a.comparar[0]), np.load(a.comparar[1])
    ia, ib = A["img"], B["img"]
    print(f"positivos: {len(ia)} vs {len(ib)} | TF {A['tf']} vs {B['tf']}")
    ka = set(zip(ia.tolist(), A["anchor"].tolist())); kb = set(zip(ib.tolist(), B["anchor"].tolist()))
    solo_a, solo_b = ka - kb, kb - ka
    print(f"pares (imagen, anchor) positivos solo en A: {len(solo_a)} | solo en B: {len(solo_b)} | imágenes distintas: {len({i for i,_ in solo_a|solo_b})}")
    da = {k: (c, o) for k, c, o in zip(zip(ia.tolist(), A['anchor'].tolist()), A['cls'], A['off'])}
    db = {k: (c, o) for k, c, o in zip(zip(ib.tolist(), B['anchor'].tolist()), B['cls'], B['off'])}
    comunes = ka & kb
    dc = sum(1 for k in comunes if da[k][0] != db[k][0])
    dmax = max((float(np.abs(da[k][1] - db[k][1]).max()) for k in comunes), default=0.0)
    n_grandes = sum(1 for k in comunes if float(np.abs(da[k][1] - db[k][1]).max()) > 1e-3)
    print(f"en los {len(comunes)} comunes: clase distinta {dc} | máx |Δoffset| {dmax:.2e} | con Δ>1e-3: {n_grandes}")
    sys.exit(0)

import tensorflow as tf
import config as C, data as D
img = tf.zeros((C.IMG_SIZE, C.IMG_SIZE, 3))
I, AN, CL, OF = [], [], [], []
n = 0
for split in ("train", "val"):
    _, gts = D.listar_muestras(split)
    for g in gts:
        _, y = D._codificar_objetivos(img, tf.constant(g)); y = y.numpy()
        idx = np.where(y[:, 0] > 0)[0]
        I += [n] * len(idx); AN += idx.tolist(); CL += y[idx, 0].astype(int).tolist(); OF.append(y[idx, 1:5]); n += 1
out = Path(__file__).with_name(f"d1c_{a.etiqueta}.npz")
np.savez(out, img=np.array(I), anchor=np.array(AN), cls=np.array(CL), off=np.concatenate(OF), tf=tf.__version__)
print(f"{out} | imágenes {n} | positivos {len(I)} | TF {tf.__version__}")
