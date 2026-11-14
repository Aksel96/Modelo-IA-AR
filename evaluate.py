"""Evaluación: Precision, Recall y mAP@0.5 en val/.  Uso: python evaluate.py"""
import argparse

import numpy as np
import tensorflow as tf

import config as C
from data import dataset_solo_imagenes, listar_muestras
from model import construir_modelo_inferencia


def iou_np(caja, cajas):
    """IoU de una caja yxyx contra un array [N,4] yxyx."""
    ymin = np.maximum(caja[0], cajas[:, 0]); xmin = np.maximum(caja[1], cajas[:, 1])
    ymax = np.minimum(caja[2], cajas[:, 2]); xmax = np.minimum(caja[3], cajas[:, 3])
    inter = np.clip(ymax - ymin, 0, None) * np.clip(xmax - xmin, 0, None)
    a = (caja[2] - caja[0]) * (caja[3] - caja[1])
    b = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def predecir(modelo_path, rutas):
    """Devuelve lista por imagen de detecciones (clase, score, yxyx) tras NMS por clase."""
    modelo = tf.keras.models.load_model(modelo_path, compile=False)
    inf = construir_modelo_inferencia(modelo)
    cajas, scores = inf.predict(dataset_solo_imagenes(rutas), verbose=0)
    todas = []
    for i in range(len(rutas)):
        dets = []
        for c in range(1, C.NUM_CLASES_CON_FONDO):
            s = scores[i, :, c]
            m = s > 0.01
            if not m.any():
                continue
            idx = tf.image.non_max_suppression(cajas[i][m], s[m], 50, C.UMBRAL_NMS).numpy()
            for j in idx:
                dets.append((c - 1, float(s[m][j]), cajas[i][m][j]))
        todas.append(dets)
    return todas


def gt_a_yxyx(g):
    return np.array([g[2] - g[4] / 2, g[1] - g[3] / 2, g[2] + g[4] / 2, g[1] + g[3] / 2])


def evaluar(modelo_path, umbral_iou=0.5, umbral_score=C.UMBRAL_SCORE):
    rutas, gts = listar_muestras("val")
    preds = predecir(modelo_path, rutas)

    aps, tp_tot, fp_tot, gt_tot = [], 0, 0, 0
    for c in range(C.NUM_CLASES):
        gt_c = {i: np.array([gt_a_yxyx(g) for g in gts[i] if g[0] == c]).reshape(-1, 4)
                for i in range(len(rutas))}
        n_gt = sum(len(v) for v in gt_c.values())
        usados = {i: np.zeros(len(v), bool) for i, v in gt_c.items()}
        dets = sorted([(s, i, b) for i, d in enumerate(preds) for (cc, s, b) in d if cc == c],
                      key=lambda t: -t[0])
        tp = np.zeros(len(dets)); scs = np.array([d[0] for d in dets])
        for k, (s, i, b) in enumerate(dets):       # emparejado greedy por score
            if len(gt_c[i]):
                ious = iou_np(b, gt_c[i])
                ious[usados[i]] = -1
                j = int(np.argmax(ious))
                if ious[j] >= umbral_iou:
                    usados[i][j] = True
                    tp[k] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        rec = ctp / max(n_gt, 1)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        # AP = área bajo la curva P-R (interpolación de todos los puntos).
        mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
        for k in range(len(mpre) - 2, -1, -1):
            mpre[k] = max(mpre[k], mpre[k + 1])
        ap = float(np.sum((mrec[1:] - mrec[:-1])[mrec[1:] != mrec[:-1]] *
                          mpre[1:][mrec[1:] != mrec[:-1]])) if n_gt else 0.0
        aps.append(ap)
        sel = scs >= umbral_score
        tp_tot += int(tp[sel].sum()); fp_tot += int((1 - tp[sel]).sum()); gt_tot += n_gt
        print(f"  AP@0.5 {C.CLASES[c]:<12}: {ap:.3f}  (GT={n_gt})")

    precision = tp_tot / max(tp_tot + fp_tot, 1)
    recall = tp_tot / max(gt_tot, 1)
    print(f"\nPrecision (score>={umbral_score}): {precision:.3f}")
    print(f"Recall    (score>={umbral_score}): {recall:.3f}")
    print(f"mAP@{umbral_iou}: {np.mean(aps):.3f}")
    return precision, recall, float(np.mean(aps))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(C.MEJOR_MODELO))
    ap.add_argument("--score", type=float, default=C.UMBRAL_SCORE)
    a = ap.parse_args()
    evaluar(a.modelo, umbral_score=a.score)
