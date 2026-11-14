"""Verifica el dataset YOLO: formato, rangos, clases y balance.  Uso: python verificar_dataset.py"""
import config as C

EXT = (".jpg", ".jpeg", ".png", ".bmp")
errores = 0
for split in ("train", "val"):
    carpeta_img = C.DATA_DIR / split / "images"
    imgs = [p for p in carpeta_img.iterdir() if p.suffix.lower() in EXT] if carpeta_img.exists() else []
    conteo, vacias = [0] * C.NUM_CLASES, 0
    for p in imgs:
        txt = C.DATA_DIR / split / "labels" / (p.stem + ".txt")
        lineas = [l for l in txt.read_text().splitlines() if l.strip()] if txt.exists() else []
        vacias += not lineas
        for l in lineas:
            v = l.split()
            try:
                assert len(v) == 5, "debe tener 5 valores"
                c = int(float(v[0]))
                assert 0 <= c < C.NUM_CLASES, f"clase {c} fuera de rango"
                assert all(0 <= float(x) <= 1 for x in v[1:]), "valores fuera de 0..1"
                conteo[c] += 1
            except (AssertionError, ValueError) as e:
                errores += 1
                print(f"[ERROR] {txt}: {e} -> '{l}'")
    print(f"{split}: {len(imgs)} imágenes, {vacias} sin objetos | " + dict(zip(C.CLASES, conteo)).__str__())
print("Dataset OK" if not errores else f"{errores} errores encontrados")
