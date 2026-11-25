"""Diagnóstico 2 (solo lectura): comportamiento real de _random_zoom y descarte de objetos visibles.
Réplica instrumentada (devuelve x0,y0,s) validada contra data._random_zoom con la misma semilla."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.getcwd())
import numpy as np, tensorflow as tf
import config as C, data as D

out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

def zoom_instrumentado(img, gt):
    """Copia literal de la rama _aplicar de data._random_zoom, devolviendo además (x0, y0, s) y el motivo de descarte."""
    s = tf.random.uniform((), 0.60, 0.90)
    valido = gt[:, 0] >= 0
    indices_validos = tf.where(valido)[:, 0]
    n_val = tf.shape(indices_validos)[0]
    def con_objetos():
        idx = tf.random.uniform((), 0, n_val, dtype=tf.int32)
        k = tf.gather(indices_validos, idx)
        obj_cx = gt[k, 1]; obj_cy = gt[k, 2]
        x0 = tf.clip_by_value(obj_cx - s * tf.random.uniform((), 0.2, 0.8), 0.0, 1.0 - s)
        y0 = tf.clip_by_value(obj_cy - s * tf.random.uniform((), 0.2, 0.8), 0.0, 1.0 - s)
        return x0, y0
    def sin_objetos():
        return tf.random.uniform((), 0.0, 1.0 - s), tf.random.uniform((), 0.0, 1.0 - s)
    x0, y0 = tf.cond(n_val > 0, con_objetos, sin_objetos)
    x1, y1 = x0 + s, y0 + s
    img_crop = tf.image.crop_and_resize(img[None, ...], [[y0, x0, y1, x1]], [0], (C.IMG_SIZE, C.IMG_SIZE))[0]
    cx_new = (gt[:, 1] - x0) / s; cy_new = (gt[:, 2] - y0) / s
    w_new = gt[:, 3] / s; h_new = gt[:, 4] / s
    xmin = tf.maximum(cx_new - w_new / 2.0, 0.0); xmax = tf.minimum(cx_new + w_new / 2.0, 1.0)
    ymin = tf.maximum(cy_new - h_new / 2.0, 0.0); ymax = tf.minimum(cy_new + h_new / 2.0, 1.0)
    w_clip = tf.maximum(xmax - xmin, 0.0); h_clip = tf.maximum(ymax - ymin, 0.0)
    cx_clip = (xmin + xmax) / 2.0; cy_clip = (ymin + ymax) / 2.0
    queda_dentro = (gt[:, 1] >= x0) & (gt[:, 1] <= x1) & (gt[:, 2] >= y0) & (gt[:, 2] <= y1)
    sigue_valido = valido & queda_dentro & (w_clip >= 0.02) & (h_clip >= 0.02)
    cls = tf.where(sigue_valido, gt[:, 0], -1.0); cx = tf.where(sigue_valido, cx_clip, -1.0)
    cy = tf.where(sigue_valido, cy_clip, -1.0); w = tf.where(sigue_valido, w_clip, -1.0); h = tf.where(sigue_valido, h_clip, -1.0)
    return img_crop, tf.stack([cls, cx, cy, w, h], axis=1), x0, y0, s, queda_dentro

rutas, gts = D.listar_muestras("train")
rutas_v, gts_v = D.listar_muestras("val")
rng = np.random.default_rng(0)
img = tf.constant(rng.random((C.IMG_SIZE, C.IMG_SIZE, 3), dtype=np.float32))

# --- 1) Validación de la réplica contra el código real (misma semilla, p=1.0 => siempre aplica el zoom)
iguales = 0; N = 400
for k in range(N):
    gt = tf.constant(gts[k % len(gts)])
    tf.random.set_seed(1000 + k)
    r_img, r_gt = D._random_zoom(img, gt, p=1.0)               # el real consume un uniform interno (hacer_zoom)
    tf.random.set_seed(1000 + k); _ = tf.random.uniform(())      # la réplica lo consume aquí para igualar la secuencia
    i_img, i_gt, *_ = zoom_instrumentado(img, gt)
    iguales += int(np.allclose(r_gt.numpy(), i_gt.numpy(), atol=1e-6) and np.allclose(r_img.numpy(), i_img.numpy(), atol=1e-5))
P(f"Validación de la réplica instrumentada frente a data._random_zoom (p=1.0, semilla igual): {iguales}/{N} casos idénticos (cajas e imagen).")
# Nota: con la misma semilla global, las operaciones aleatorias en eager comparten secuencia por contador de ops;
# si ambas coinciden, el instrumento reproduce el código real.

# --- 2) Estadísticas sobre todo el set de train (y val como referencia), varias pasadas con p=1.0 (zoom siempre aplicado)
def estadisticas(gts_set, pasadas, etiqueta):
    tf.random.set_seed(7)
    filas = []; imgs = []
    for p_ in range(pasadas):
        for i in range(len(gts_set)):
            gt = gts_set[i]; v = np.where(gt[:, 0] >= 0)[0]
            _, g2, x0, y0, s, dentro = zoom_instrumentado(img, tf.constant(gt))
            g2 = g2.numpy(); x0, y0, s = float(x0), float(y0), float(s); dentro = dentro.numpy()
            todos_perdidos_visibles = False; n_vis50_perd = 0
            for j in v:
                cx, cy, w, h = gt[j, 1:5]
                ix = max(0.0, min(cx + w/2, x0 + s) - max(cx - w/2, x0)); iy = max(0.0, min(cy + h/2, y0 + s) - max(cy - h/2, y0))
                vis = (ix * iy) / (w * h)
                kept = g2[j, 0] >= 0
                # visibilidad relativa del lado: ancho/alto clipeado respecto al original
                filas.append((int(gt[j, 0]), vis, kept, bool(dentro[j]), w / s, h / s, g2[j, 3] if kept else np.nan, g2[j, 4] if kept else np.nan, w, h))
                if (not kept) and vis >= 0.5: n_vis50_perd += 1
            imgs.append((len(v), int((g2[:, 0] >= 0).sum()), n_vis50_perd, int(gt[v[0], 0])))
    F = np.array(filas, dtype=float); I = np.array(imgs)
    P(f"\n===== {etiqueta}: {len(gts_set)} imágenes × {pasadas} pasadas con zoom aplicado (p=1.0) = {len(F)} GT-observaciones =====")
    P(f"{'clase':<10}{'obs':>7}{'descartado':>11}{'desc. con vis≥30%':>18}{'vis≥50%':>9}{'vis≥70%':>9}{'cent. fuera':>12}{'cent.dentro y <0.02':>20}{'conservado vis<50%':>19}{'conserv. vis<80%':>17}")
    for c, n in enumerate(C.CLASES):
        m = F[:, 0] == c; f = F[m]; tot = len(f)
        desc = f[:, 2] == 0
        P(f"{n:<10}{tot:>7}{100*desc.mean():>10.1f}%{100*(desc & (f[:,1] >= .3)).mean():>17.1f}%{100*(desc & (f[:,1] >= .5)).mean():>8.1f}%"
          f"{100*(desc & (f[:,1] >= .7)).mean():>8.1f}%{100*(f[:,3] == 0).mean():>11.1f}%{100*(desc & (f[:,3] == 1)).mean():>19.1f}%"
          f"{100*((~desc) & (f[:,1] < .5)).mean():>18.1f}%{100*((~desc) & (f[:,1] < .8)).mean():>16.1f}%")
    m = np.ones(len(F), bool); f = F; desc = f[:, 2] == 0
    P(f"{'TOTAL':<10}{len(f):>7}{100*desc.mean():>10.1f}%{100*(desc & (f[:,1] >= .3)).mean():>17.1f}%{100*(desc & (f[:,1] >= .5)).mean():>8.1f}%{100*(desc & (f[:,1] >= .7)).mean():>8.1f}%")
    P("\nPor imagen (con zoom aplicado): ")
    for c, n in enumerate(C.CLASES):
        m = I[:, 3] == c
        P(f"  {n:<10} imágenes-pasada {m.sum():>5} | se queda sin NINGÚN objeto: {100*(I[m,1]==0).mean():5.1f}% | "
          f"pierde ≥1 objeto con ≥50% visible: {100*(I[m,2]>0).mean():5.1f}% | objetos conservados / originales: {I[m,1].sum()/I[m,0].sum():.2f}")
    P("\nEscala efectiva (caja conservada, ancho y alto del GT en 256 px tras el zoom; mediana) vs sin zoom:")
    for c, n in enumerate(C.CLASES):
        f = F[F[:, 0] == c]; k = f[:, 2] == 1
        P(f"  {n:<10} sin zoom {np.median(f[:,8])*256:6.1f}×{np.median(f[:,9])*256:6.1f} px | con zoom (conservadas) {np.nanmedian(f[k,6])*256:6.1f}×{np.nanmedian(f[k,7])*256:6.1f} px")
    return F

F = estadisticas(gts, 5, "TRAIN")
estadisticas(gts_v, 2, "VAL (solo referencia; el zoom NO se aplica en val)")

# --- 3) Qué ocurre con p=0.5 real en una época: fracción de GT-observaciones afectadas
P("\nEn entrenamiento el zoom se aplica con p=0.5: la proporción de GT perdidos por época es la mitad de la de arriba "
  "(p. ej. descarte total ≈ {:.1f}%).".format(50 * (F[:, 2] == 0).mean()))
open(os.path.join(os.path.dirname(__file__), "d2_zoom_salida.txt"), "w", encoding="utf-8").write("\n".join(out))
