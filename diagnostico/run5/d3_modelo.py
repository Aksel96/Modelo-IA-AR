"""Diagnóstico 3 (SOLO INFERENCIA, no entrena): val_loss frente a mAP para un checkpoint.

Uso (desde la raíz del repo, con la MISMA bandera EXIF con la que se entrenó el modelo):
    APLICAR_EXIF=1 python diagnostico/run5/d3_modelo.py --modelo archivo/best_m5.keras --etiqueta run5
    python diagnostico/run5/d3_modelo.py --modelo /tmp/run4.keras --etiqueta run4          # Run 4: sin EXIF

Qué mide (val y una muestra de train SIN aumentos):
  1. La pérdida de entrenamiento descompuesta (clasificación en positivos, negativos minados, regresión),
     por clase. Se comprueba contra losses.perdida_ssd (debe coincidir; en val = val_loss del historial).
  2. Probabilidad de la clase correcta en los anchors positivos y falsos positivos entre los negativos.
  3. mAP@0.5, precisión y recall a varios umbrales de score, AP "agnóstico de clase" (localización pura),
     recall potencial (GT con alguna detección IoU>=0.5) y matriz de confusión de clase.
No modifica archivos del proyecto: escribe solo en --salida.
"""
import argparse, os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

import numpy as np
import tensorflow as tf

import config as C
import data as D
import evaluate as E
from losses import perdida_ssd, smooth_l1

NIVEL = np.concatenate([np.full(f * f * 6, k) for k, f in enumerate(C.FEATURE_MAPS)])
SALIDA = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    SALIDA.append(s)


def componentes(y_true, logits_box):
    """Réplica por imagen de losses.perdida_ssd, devolviendo sus tres sumandos y el nº de positivos."""
    cls_true = tf.cast(y_true[..., 0], tf.int32)
    pos = cls_true > 0
    pos_f = tf.cast(pos, tf.float32)
    n_pos = tf.reduce_sum(pos_f, axis=1)
    logits = logits_box[..., 4:]
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cls_true, logits=logits)
    ce_neg = tf.where(pos, 0.0, ce)
    rango = tf.argsort(tf.argsort(ce_neg, axis=1, direction="DESCENDING"), axis=1)
    n_neg = tf.maximum(C.RATIO_NEGATIVOS * n_pos, 10.0)
    neg_mask = tf.cast(tf.cast(rango, tf.float32) < n_neg[:, None], tf.float32) * (1.0 - pos_f)
    ce_pos = tf.reduce_sum(ce * pos_f, axis=1)
    ce_negm = tf.reduce_sum(ce * neg_mask, axis=1)
    box = tf.reduce_sum(tf.reduce_sum(smooth_l1(y_true[..., 1:5] - logits_box[..., :4]), axis=-1) * pos_f, axis=1)
    return ce_pos.numpy(), ce_negm.numpy(), box.numpy(), n_pos.numpy()


def medir_perdida(modelo, split, max_lotes=None):
    ds = D.construir_dataset(split, entrenamiento=False)
    if max_lotes:
        ds = ds.take(max_lotes)
    lotes = []                       # (tamaño, loss_ref, loss_mia)
    imgs = []                        # por imagen: clase, ce_pos, ce_neg, box, n_pos
    pos_stats = {c: dict(p_true=[], ok=[], p_bg=[], niv=[]) for c in range(C.NUM_CLASES)}
    neg_fp = []                      # por imagen: anchors negativos con prob de objeto > 0.5
    for x, y in ds:
        pred = modelo(x, training=False)
        ref = float(perdida_ssd(y, pred))
        cp, cn, bx, npos = componentes(y, pred)
        tot = (cp.sum() + cn.sum() + bx.sum()) / max(npos.sum(), 1.0)
        lotes.append((int(x.shape[0]), ref, float(tot)))
        probs = tf.nn.softmax(pred[..., 4:], axis=-1).numpy()
        yc = y[..., 0].numpy().astype(int)
        for b in range(x.shape[0]):
            pos = yc[b] > 0
            if pos.any():
                c = int(np.bincount(yc[b][pos]).argmax()) - 1          # una sola clase por imagen
                pt = probs[b][pos, yc[b][pos]]
                pos_stats[c]["p_true"] += pt.tolist()
                pos_stats[c]["ok"] += (probs[b][pos].argmax(1) == yc[b][pos]).tolist()
                pos_stats[c]["p_bg"] += probs[b][pos, 0].tolist()
                pos_stats[c]["niv"] += NIVEL[pos].tolist()
                imgs.append((c, cp[b], cn[b], bx[b], npos[b]))
                neg_fp.append((c, int(((1 - probs[b][~pos, 0]) > 0.5).sum())))
    n = np.array([l[0] for l in lotes], float)
    ref = float(np.sum(n * [l[1] for l in lotes]) / n.sum())
    mia = float(np.sum(n * [l[2] for l in lotes]) / n.sum())
    P(f"\n===== PÉRDIDA en '{split}' (sin aumentos), {int(n.sum())} imágenes =====")
    P(f"perdida_ssd (código real, media ponderada por lote): {ref:.4f} | réplica por componentes: {mia:.4f} "
      f"| diferencia {abs(ref - mia):.2e}")
    I = np.array(imgs)
    cls, cp, cn, bx, npos = I[:, 0].astype(int), I[:, 1], I[:, 2], I[:, 3], I[:, 4]
    P(f"Componentes globales por positivo: cls_pos {cp.sum()/npos.sum():.3f} + cls_neg(minados) {cn.sum()/npos.sum():.3f} "
      f"+ caja {bx.sum()/npos.sum():.3f} = {(cp.sum()+cn.sum()+bx.sum())/npos.sum():.3f}")
    P(f"{'clase':<10}{'img':>5}{'pos':>6}{'cls_pos':>9}{'cls_neg':>9}{'caja':>8}{'total':>8} | {'p(clase)':>9}{'acierto':>9}{'p(fondo)':>9}{'FP-neg/img':>11}")
    for c, nombre in enumerate(C.CLASES):
        m = cls == c
        if not m.any():
            continue
        s = pos_stats[c]; fp = [v for cc, v in neg_fp if cc == c]
        P(f"{nombre:<10}{m.sum():>5}{int(npos[m].sum()):>6}{cp[m].sum()/npos[m].sum():>9.3f}{cn[m].sum()/npos[m].sum():>9.3f}"
          f"{bx[m].sum()/npos[m].sum():>8.3f}{(cp[m].sum()+cn[m].sum()+bx[m].sum())/npos[m].sum():>8.3f} | "
          f"{np.mean(s['p_true']):>9.3f}{np.mean(s['ok']):>9.3f}{np.mean(s['p_bg']):>9.3f}{np.mean(fp):>11.1f}")
    return ref


def ap_voc(tp, scs, n_gt):
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    o = np.argsort(-scs); tp = tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    d = mrec[1:] != mrec[:-1]
    return float(np.sum((mrec[1:] - mrec[:-1])[d] * mpre[1:][d]))


def medir_deteccion(modelo_path, split, paso, max_img=None):
    rutas, gts = D.listar_muestras(split)
    idx = list(range(0, len(rutas), paso))[:max_img]
    rutas = [rutas[i] for i in idx]; gts = gts[idx]
    preds = E.predecir(modelo_path, rutas)
    P(f"\n===== DETECCIÓN en '{split}' (cada {paso}ª imagen, {len(rutas)} imágenes, NMS por clase como evaluate.py) =====")
    gt_por_img = {i: [(int(g[0]), E.gt_a_yxyx(g)) for g in gts[i] if g[0] >= 0] for i in range(len(rutas))}
    n_gt_c = [sum(1 for i in gt_por_img for (c, _) in gt_por_img[i] if c == k) for k in range(C.NUM_CLASES)]

    # AP por clase (misma fórmula que evaluate.py) y AP agnóstico de clase
    aps = []
    for c in range(C.NUM_CLASES):
        gc = {i: np.array([b for (k, b) in gt_por_img[i] if k == c]).reshape(-1, 4) for i in gt_por_img}
        used = {i: np.zeros(len(v), bool) for i, v in gc.items()}
        dets = sorted([(s, i, b) for i, d in enumerate(preds) for (cc, s, b) in d if cc == c], key=lambda t: -t[0])
        tp = np.zeros(len(dets)); sc = np.array([d[0] for d in dets])
        for k, (s, i, b) in enumerate(dets):
            if len(gc[i]):
                io = E.iou_np(b, gc[i]); io[used[i]] = -1; j = int(np.argmax(io))
                if io[j] >= 0.5:
                    used[i][j] = True; tp[k] = 1
        aps.append(ap_voc(tp, sc, n_gt_c[c]))
    P("AP@0.5 por clase: " + ", ".join(f"{C.CLASES[c]} {aps[c]:.3f}" for c in range(C.NUM_CLASES)) + f"  | mAP {np.mean(aps):.3f}")

    ga = {i: np.array([b for (_, b) in gt_por_img[i]]).reshape(-1, 4) for i in gt_por_img}
    used = {i: np.zeros(len(v), bool) for i, v in ga.items()}
    dets = sorted([(s, i, b) for i, d in enumerate(preds) for (cc, s, b) in d], key=lambda t: -t[0])
    tp = np.zeros(len(dets)); sc = np.array([d[0] for d in dets])
    for k, (s, i, b) in enumerate(dets):
        if len(ga[i]):
            io = E.iou_np(b, ga[i]); io[used[i]] = -1; j = int(np.argmax(io))
            if io[j] >= 0.5:
                used[i][j] = True; tp[k] = 1
    P(f"AP@0.5 AGNÓSTICO de clase (solo localización/objectness): {ap_voc(tp, sc, sum(n_gt_c)):.3f}")

    # Precisión/recall por umbral (emparejado por clase, como evaluate.py)
    P(f"{'score>=':>8}{'detecciones':>13}{'TP':>6}{'precisión':>11}{'recall':>8}")
    for u in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02):
        tpt = fpt = 0
        for c in range(C.NUM_CLASES):
            gc = {i: np.array([b for (k, b) in gt_por_img[i] if k == c]).reshape(-1, 4) for i in gt_por_img}
            used_ = {i: np.zeros(len(v), bool) for i, v in gc.items()}
            dd = sorted([(s, i, b) for i, d in enumerate(preds) for (cc, s, b) in d if cc == c and s >= u], key=lambda t: -t[0])
            for (s, i, b) in dd:
                ok = False
                if len(gc[i]):
                    io = E.iou_np(b, gc[i]); io[used_[i]] = -1; j = int(np.argmax(io))
                    if io[j] >= 0.5:
                        used_[i][j] = True; ok = True
                tpt += ok; fpt += (not ok)
        P(f"{u:>8}{tpt+fpt:>13}{tpt:>6}{tpt/max(tpt+fpt,1):>11.3f}{tpt/max(sum(n_gt_c),1):>8.3f}")

    # Recall potencial y confusión: para cada GT, detección (cualquier clase, score>0.01) de mayor score con IoU>=0.5
    conf = np.zeros((C.NUM_CLASES, C.NUM_CLASES + 1), int); mejor_iou = [[] for _ in range(C.NUM_CLASES)]
    for i in range(len(rutas)):
        for (c, b) in gt_por_img[i]:
            cand = [(s, cc, E.iou_np(b, d_b[None])[0]) for (cc, s, d_b) in preds[i]]
            mejor_iou[c].append(max([x[2] for x in cand], default=0.0))
            ok = [x for x in cand if x[2] >= 0.5]
            if ok:
                conf[c, max(ok, key=lambda x: x[0])[1]] += 1
            else:
                conf[c, C.NUM_CLASES] += 1
    P("\nPara cada GT: clase de la detección de mayor score (score>0.01, tras NMS) con IoU>=0.5; última columna = ninguna.")
    P("GT / detectada".ljust(16) + "".join(f"{n:>10}" for n in C.CLASES) + f"{'ninguna':>10}{'IoU máx med.':>14}")
    for c, nombre in enumerate(C.CLASES):
        P(f"{nombre:<16}" + "".join(f"{v:>10}" for v in conf[c]) + f"{np.median(mejor_iou[c]) if mejor_iou[c] else 0:>14.2f}")
    return aps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="aleatorio", help="ruta .keras, o 'aleatorio' (modelo sin entrenar, solo para probar el script)")
    ap.add_argument("--etiqueta", default="modelo")
    ap.add_argument("--salida", default=str(RAIZ / "diagnostico" / "run5"))
    ap.add_argument("--rapido", action="store_true", help="prueba del script con pocos lotes")
    a = ap.parse_args()
    P(f"Etiqueta: {a.etiqueta} | modelo: {a.modelo} | APLICAR_EXIF={C.APLICAR_EXIF} | TF {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")
    if a.modelo == "aleatorio":
        from model import construir_ssd
        modelo = construir_ssd(); path = None
        tmp = Path(a.salida) / "_modelo_aleatorio.keras"; modelo.save(tmp); path = str(tmp)
    else:
        path = a.modelo
        modelo = tf.keras.models.load_model(path, compile=False)
    lotes = 3 if a.rapido else None
    medir_perdida(modelo, "val", lotes)
    medir_perdida(modelo, "train", lotes)
    medir_deteccion(path, "val", 1 if not a.rapido else 40)
    medir_deteccion(path, "train", 4 if not a.rapido else 100)
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    out = Path(a.salida) / f"d3_{a.etiqueta}.txt"
    out.write_text("\n".join(SALIDA), encoding="utf-8")
    print(f"\nEscrito: {out}")
    if a.modelo == "aleatorio":
        (Path(a.salida) / "_modelo_aleatorio.keras").unlink(missing_ok=True)
