"""Diagnóstico 5 (solo lectura, sin modelo ni entrenamiento): cobertura de anchors si el umbral de IoU positivo fuera otro.
No cambia config.py: calcula con anchors.py y las etiquetas, para dimensionar la propuesta E5a antes de pedir autorización."""
import os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, tensorflow as tf
import config as C, data as D
from anchors import cxcywh_a_yxyx, matriz_iou

out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

rutas, gts = D.listar_muestras("train")
filas = []
for g in gts:
    v = np.where(g[:, 0] >= 0)[0]
    iou = matriz_iou(D._ANCHORS_YXYX, cxcywh_a_yxyx(tf.constant(g[v, 1:5]))).numpy()      # [A, n]
    for k, j in enumerate(v):
        filas.append((int(g[j, 0]), iou[:, k]))
P("train (1,097 imágenes). Por umbral: % de GT SIN ningún anchor con IoU>=umbral (se les fuerza solo su mejor anchor), "
  "media de anchors por GT, y % de GT con <=2.")
for u in (0.5, 0.45, 0.4, 0.35, 0.3):
    P(f"\n--- umbral {u} ---")
    P(f"{'clase':<10}{'GT':>5}{'sin anchor':>12}{'≤2 anchors':>12}{'media':>8}")
    tot_pos = 0
    for c, n in enumerate(C.CLASES):
        f = [r for r in filas if r[0] == c]
        cnt = np.array([(r[1] >= u).sum() for r in f])
        P(f"{n:<10}{len(f):>5}{100*np.mean(cnt==0):>11.1f}%{100*np.mean(cnt<=2):>11.1f}%{cnt.mean():>8.1f}")
        tot_pos += cnt.sum()
    P(f"anchors positivos por imagen (suma de GT, antes de colisiones): {tot_pos/len(gts):.1f}")
open(Path(__file__).with_name("d5_umbral_iou.txt"), "w", encoding="utf-8").write("\n".join(out))
