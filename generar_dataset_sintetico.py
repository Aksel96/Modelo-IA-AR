"""Genera un dataset YOLO sintético (rectángulos de colores) para probar el pipeline.

Uso: python generar_dataset_sintetico.py [--n 60]
NO sirve para entrenar un detector real; solo es un smoke test.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw

import config as C

COLORES = [(230, 200, 20), (40, 40, 40), (200, 30, 30), (30, 120, 220), (30, 180, 60)]


def generar(split, n, rng):
    (C.DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (C.DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (320, 320), tuple(int(v) for v in rng.integers(150, 255, 3)))
        d = ImageDraw.Draw(img)
        lineas = []
        for _ in range(int(rng.integers(1, 4))):
            c = int(rng.integers(0, C.NUM_CLASES))
            w, h = rng.uniform(0.15, 0.4, 2)
            cx, cy = rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2)
            d.rectangle([(cx - w / 2) * 320, (cy - h / 2) * 320,
                         (cx + w / 2) * 320, (cy + h / 2) * 320], fill=COLORES[c % 5])
            lineas.append(f"{c} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}")
        img.save(C.DATA_DIR / split / "images" / f"{i:04d}.jpg")
        (C.DATA_DIR / split / "labels" / f"{i:04d}.txt").write_text("\n".join(lineas) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    n = ap.parse_args().n
    rng = np.random.default_rng(0)
    generar("train", n, rng)
    generar("val", max(n // 4, 8), rng)
    print("Dataset sintético creado en", C.DATA_DIR)
