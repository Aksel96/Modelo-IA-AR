"""Diagnóstico 1 (solo lectura): asignación de anchors, colisiones entre GT y positivos por clase.
Usa las funciones reales de data.py/anchors.py. No modifica nada."""
import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.getcwd())
import numpy as np, tensorflow as tf
import config as C, data as D
from anchors import cxcywh_a_yxyx, matriz_iou, ANCHORS_CXCYWH

A = ANCHORS_CXCYWH.shape[0]
NIVEL = np.concatenate([np.full(f * f * 6, k) for k, f in enumerate(C.FEATURE_MAPS)])   # nivel de cada anchor
TIPO = np.concatenate([np.tile(np.arange(6), f * f) for f in C.FEATURE_MAPS])           # 0..5 ratio
TIPOS = ["1:1", "1:1'", "2:1", "1:2", "3:1", "1:3"]
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

def referencia(gt, variante):
    """Réplica numpy de la asignación; variante 'ultimo'/'primero' = quién gana con índices repetidos."""
    valido = gt[:, 0] >= 0
    iou = matriz_iou(D._ANCHORS_YXYX, cxcywh_a_yxyx(tf.constant(gt[:, 1:5]))).numpy()
    iou = np.where(valido[None, :], iou, -1.0)
    mejor_gt = iou.argmax(1); mejor_iou = iou.max(1)
    nat_pos = (iou >= C.UMBRAL_IOU_POSITIVO)                      # [A,M]
    ids = np.where(valido)[0]
    mejor_anchor = {j: int(iou[:, j].argmax()) for j in ids}
    orden = ids if variante == "ultimo" else ids[::-1]
    for j in orden:                                               # la última escritura gana
        a = mejor_anchor[j]; mejor_iou[a] = 2.0; mejor_gt[a] = j
    positivo = mejor_iou >= C.UMBRAL_IOU_POSITIVO
    return iou, mejor_gt, positivo, nat_pos, mejor_anchor

def analizar(split):
    rutas, gts = D.listar_muestras(split)
    n = len(rutas)
    coincide = {"ultimo": 0, "primero": 0}; imgs_colision = 0
    filas = []   # por GT
    pos_por_img = []
    for i in range(n):
        gt = gts[i]
        img = tf.zeros((C.IMG_SIZE, C.IMG_SIZE, 3))
        _, y = D._codificar_objetivos(img, tf.constant(gt))
        y = y.numpy()
        res = {}
        for v in ("ultimo", "primero"):
            iou, mg, pos, nat, ma = referencia(gt, v)
            cls = np.where(pos, gt[mg, 0] + 1, 0)
            res[v] = (iou, mg, pos, nat, ma, cls)
            cajas = D.codificar(tf.constant(gt[mg, 1:5]), D._ANCHORS).numpy()
            cajas = np.where(pos[:, None], cajas, 0.0)
            if np.array_equal(cls, y[:, 0]) and np.allclose(cajas, y[:, 1:], atol=1e-5): coincide[v] += 1
        iou, mg, pos, nat, ma, cls = res["ultimo"]          # (se informa abajo cuál coincide con el código real)
        valido = np.where(gt[:, 0] >= 0)[0]
        pos_por_img.append(int(pos.sum()))
        anchors_forzados = [ma[j] for j in valido]
        hay_col = len(set(anchors_forzados)) < len(anchors_forzados)
        imgs_colision += hay_col
        # anclas con IoU>=0.5 con >=2 GT
        multi = int(((nat[:, valido]).sum(1) >= 2).sum()) if len(valido) else 0
        for j in valido:
            comparte = sum(1 for k in valido if k != j and ma[k] == ma[j])
            # anchors finalmente positivos y asignados a j según el código real (clase+offsets): usamos la referencia que coincide
            n_final = int(((mg == j) & pos).sum())
            filas.append(dict(split=split, img=i, clase=int(gt[j, 0]), n_nat=int(nat[:, j].sum()), max_iou=float(iou[:, j].max()),
                              comparte=comparte, n_final=n_final, w=float(gt[j, 3]), h=float(gt[j, 4]),
                              mejor_anchor_nivel=int(NIVEL[ma[j]]), mejor_anchor_tipo=int(TIPO[ma[j]]),
                              n_gt_img=len(valido), multi_img=multi))
    return n, coincide, imgs_colision, filas, pos_por_img

res_all = {}
for split in ("train", "val"):
    n, coincide, ic, filas, ppi = analizar(split)
    res_all[split] = filas
    P(f"\n===== {split}: {n} imágenes, {len(filas)} GT =====")
    P(f"La referencia numpy reproduce EXACTAMENTE las clases asignadas por el código real en: "
      f"'último gana' {coincide['ultimo']}/{n} imágenes, 'primero gana' {coincide['primero']}/{n}")
    P(f"Imágenes con colisión (dos GT con el mismo mejor anchor): {ic}  | positivos por imagen: media {np.mean(ppi):.1f}, min {min(ppi)}, max {max(ppi)}; "
      f"imágenes con 0 positivos: {sum(1 for p in ppi if p==0)}")
    P(f"{'clase':<10}{'GT':>5}{'IoU max<0.5':>12}{'sin anchor nat.':>16}{'≤2 nat.':>9}{'nat. media':>11}{'comparte':>9}{'sin ancla final':>16}{'final media':>12}")
    for c, nombre in enumerate(C.CLASES):
        f = [r for r in filas if r["clase"] == c]
        g = len(f)
        P(f"{nombre:<10}{g:>5}{100*np.mean([r['max_iou']<0.5 for r in f]):>11.1f}%{100*np.mean([r['n_nat']==0 for r in f]):>15.1f}%"
          f"{100*np.mean([r['n_nat']<=2 for r in f]):>8.1f}%{np.mean([r['n_nat'] for r in f]):>11.1f}"
          f"{100*np.mean([r['comparte']>0 for r in f]):>8.1f}%{100*np.mean([r['n_final']==0 for r in f]):>15.1f}%{np.mean([r['n_final'] for r in f]):>12.1f}")

# segunda pasada: positivos por clase / nivel / tipo (referencia validada), sin y con aumento (zoom real)
def pos_por_clase(split, usar_zoom, veces, semilla=0):
    rutas, gts = D.listar_muestras(split)
    tf.random.set_seed(semilla)
    por_nivel = np.zeros((C.NUM_CLASES, 5), int); por_tipo = np.zeros((C.NUM_CLASES, 6), int); n_gt_vis = np.zeros(C.NUM_CLASES, int)
    img = tf.zeros((C.IMG_SIZE, C.IMG_SIZE, 3))
    for _ in range(veces):
        for i in range(len(rutas)):
            gt = tf.constant(gts[i])
            if usar_zoom:
                _, gt = D._random_zoom(img, gt, p=0.5)
            _, y = D._codificar_objetivos(img, gt)
            y = y.numpy(); g = gt.numpy()
            for c in range(C.NUM_CLASES):
                m = y[:, 0] == c + 1
                por_nivel[c] += np.bincount(NIVEL[m], minlength=5); por_tipo[c] += np.bincount(TIPO[m], minlength=6)
                n_gt_vis[c] += int((g[:, 0] == c).sum())
    return por_nivel, por_tipo, n_gt_vis

for split, zoom, veces in (("train", False, 1), ("train", True, 3), ("val", False, 1)):
    pn, pt, ng = pos_por_clase(split, zoom, veces)
    P(f"\n--- Anchors positivos ({split}, {'CON zoom real p=0.5, %d pasadas' % veces if zoom else 'sin aumentos'}); GT visibles totales = {ng.tolist()}")
    P(f"{'clase':<10}" + "".join(f"{'L'+str(k):>8}" for k in range(5)) + f"{'total':>8}{'pos/GT':>8}   " + "  ".join(f"{t:>5}" for t in TIPOS))
    for c, nombre in enumerate(C.CLASES):
        P(f"{nombre:<10}" + "".join(f"{v:>8}" for v in pn[c]) + f"{pn[c].sum():>8}{pn[c].sum()/max(ng[c],1):>8.1f}   " + "  ".join(f"{v:>5}" for v in pt[c]))
    P(f"Total positivos {int(pn.sum())}; anchors totales evaluados {A} por imagen.")

# Tamaños de GT de los GT sin ningún anchor natural
P("\n--- GT sin anchor natural (IoU<0.5 con todos): tamaño mediano de caja (w,h normalizados) y mejor anchor forzado")
for c, nombre in enumerate(C.CLASES):
    f = [r for r in res_all["train"] + res_all["val"] if r["clase"] == c and r["n_nat"] == 0]
    if f:
        P(f"{nombre:<10} n={len(f):>4}  w̃={np.median([r['w'] for r in f]):.3f} h̃={np.median([r['h'] for r in f]):.3f}  "
          f"IoU del mejor anchor: mediana {np.median([r['max_iou'] for r in f]):.2f}  nivel del mejor anchor: {np.bincount([r['mejor_anchor_nivel'] for r in f], minlength=5).tolist()}")
open(os.path.join(os.path.dirname(__file__), "d1_anchors_salida.txt"), "w", encoding="utf-8").write("\n".join(out))
