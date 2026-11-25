"""Diagnóstico 4 (solo lectura): curvas de aprendizaje del Run 5 a partir de archivo/historial_m5.csv."""
import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parents[2]
f = list(csv.DictReader(open(RAIZ / "archivo" / "historial_m5.csv")))
ep = np.array([int(r["epoch"]) + 1 for r in f]); lr = np.array([float(r["learning_rate"]) for r in f])
tr = np.array([float(r["loss"]) for r in f]); va = np.array([float(r["val_loss"]) for r in f])
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

k = int(va.argmin()); mn = va[k]
P(f"Épocas: {len(ep)} | mejor época {ep[k]} (val_loss {mn:.4f}, loss {tr[k]:.4f}) | final: loss {tr[-1]:.4f}, val_loss {va[-1]:.4f}")
def primera(umbral):
    i = np.where(va[k:] > mn * umbral)[0]
    return int(ep[k + i[0]]) if len(i) else None
P("val_loss respecto al mínimo (época 8), primera época posterior que lo supera en: " +
  ", ".join(f"+{int(round((u-1)*100))}%: {primera(u)}" for u in (1.02, 1.10, 1.25, 1.50, 1.75)))
P(f"Épocas con val_loss < mínimo+0.5 antes de la 8: {int((va[:k] < mn+0.5).sum())}")
P("\nBrecha val_loss/loss (la loss de train lleva aumentos, la de val no): ")
for e in (1, 4, 8, 9, 10, 13, 18, 23, 28, 33, 38):
    P(f"  época {e:>2}: loss {tr[e-1]:.3f} | val_loss {va[e-1]:.3f} | cociente {va[e-1]/tr[e-1]:.2f} | diferencia {va[e-1]-tr[e-1]:+.2f} | LR {lr[e-1]:.3g}")
P("\nVelocidad de cambio (promedio por época):")
for a, b, nombre in ((1, 8, "épocas 1→8"), (8, 13, "épocas 8→13 (LR 1e-3)"), (13, 23, "épocas 13→23 (LR 5e-4..2.5e-4)"), (23, 38, "épocas 23→38 (LR ≤1.25e-4)")):
    P(f"  {nombre:<32} loss {(tr[b-1]-tr[a-1])/(b-a):+.3f}/época | val_loss {(va[b-1]-va[a-1])/(b-a):+.3f}/época")
P(f"\nSubidas de val_loss entre épocas consecutivas tras la 8: {int((np.diff(va[k:]) > 0).sum())} de {len(va)-k-1}; "
  f"bajadas: {int((np.diff(va[k:]) < 0).sum())} (épocas {[int(ep[k+1+i]) for i in np.where(np.diff(va[k:]) < 0)[0]]})")
P(f"Correlación (épocas 8→38) loss vs val_loss: {np.corrcoef(tr[k:], va[k:])[0,1]:+.3f}  (loss baja, val sube)")
P("\nLR: la val_loss ya había subido +{:.0f}% sobre el mínimo (época 13) cuando ReduceLROnPlateau actuó por primera vez; "
  "con patience=5 la primera reducción solo puede ocurrir 5 épocas sin mejora después de la mejor (8+5=13).".format(100*(va[12]/mn-1)))
P("Tras la primera reducción (época 13→14) val_loss baja de 10.95 a 10.30 pero no vuelve a bajar de 10.25; el mejor checkpoint (época 8) no se supera nunca.")
P("Referencia Runs 3/4 (bitácora, sin historial en el repo): Run 3 mejor val_loss 8.1028 (~época 9), loss final 2.2294; "
  "Run 4 mejor val_loss 7.5415 (época 11), loss final 3.3623. Mismo patrón: mínimo temprano y deterioro posterior.")
open(Path(__file__).with_name("d4_curvas.txt"), "w", encoding="utf-8").write("\n".join(out))

# Gráfico sencillo (PIL, sin matplotlib): dos series, marcas en los cambios de LR y en la mejor epoca
W, H, m = 900, 520, 70
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
x = lambda e: m + (e - 1) / (len(ep) - 1) * (W - 2 * m)
ymax = 16.0; y = lambda v: H - m - v / ymax * (H - 2 * m)
for v in range(0, 17, 2):
    d.line([(m, y(v)), (W - m, y(v))], fill=(225, 225, 225)); d.text((m - 28, y(v) - 6), str(v), fill=(60, 60, 60))
for e in (1, 8, 13, 18, 23, 28, 33, 38):
    d.line([(x(e), H - m), (x(e), H - m + 5)], fill=(60, 60, 60)); d.text((x(e) - 6, H - m + 8), str(e), fill=(60, 60, 60))
cambio = [int(ep[i]) for i in range(1, len(ep)) if lr[i] < lr[i-1]]
for e in cambio:
    d.line([(x(e) - 0.5*(x(2)-x(1)), m), (x(e) - 0.5*(x(2)-x(1)), H - m)], fill=(200, 200, 200), width=1)
d.line([(x(ep[k]), m), (x(ep[k]), H - m)], fill=(120, 120, 120), width=2)
C_TR, C_VA = (0, 114, 178), (213, 94, 0)
for serie, col in ((tr, C_TR), (va, C_VA)):
    pts = [(x(e), y(v)) for e, v in zip(ep, serie)]; d.line(pts, fill=col, width=3)
    for p in pts: d.ellipse([p[0]-2.5, p[1]-2.5, p[0]+2.5, p[1]+2.5], fill=col)
d.text((x(25), y(tr[24]) - 22), "loss (train, con aumentos)", fill=C_TR); d.text((x(20), y(va[19]) - 24), "val_loss (sin aumentos)", fill=C_VA)
d.text((x(ep[k]) + 5, m + 4), f"mejor: epoca {ep[k]} (val_loss {mn:.2f})", fill=(60, 60, 60))
d.text((m, 14), "Run 5 (E1, EXIF): perdida por epoca. Lineas grises claras: reducciones de LR (patience=5)", fill=(30, 30, 30))
d.text((W // 2 - 20, H - 24), "epoca", fill=(60, 60, 60))
img.save(Path(__file__).with_name("curvas_run5.png"))
print("PNG guardado")
