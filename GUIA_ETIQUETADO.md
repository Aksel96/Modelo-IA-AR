# Guía de etiquetado de imágenes (formato YOLO)

Objetivo: dibujar una caja alrededor de cada objeto (moneda, laptop, boligrafo, raton, libro) y guardar un `.txt` por imagen con el formato que lee este proyecto.

## 1. Formato de etiqueta
Un `.txt` por imagen, **mismo nombre** (`foto001.jpg` → `foto001.txt`). Una línea por objeto:
```
clase cx cy w h
```
| Campo | Significado |
|---|---|
| `clase` | Índice: 0 moneda, 1 laptop, 2 boligrafo, 3 raton, 4 libro (orden de `CLASES` en `config.py`) |
| `cx`, `cy` | Centro de la caja ÷ ancho / alto de la imagen |
| `w`, `h` | Ancho / alto de la caja ÷ ancho / alto de la imagen |

Ejemplo (imagen 640×480, moneda en x 300–380, y 200–280):
`cx=340/640=0.531`, `cy=240/480=0.5`, `w=80/640=0.125`, `h=80/480=0.167`
```
0 0.531 0.500 0.125 0.167
```
Todos los valores deben estar entre 0 y 1. Si ves números mayores que 1, la herramienta exportó en píxeles: está mal.

## 2. Antes de etiquetar: tomar buenas fotos
- ≥ 300 fotos por clase (mejor 500+), entrenando desde cero.
- Varía: fondos, iluminación (día, noche, lámpara), distancia, ángulo, orientación, desenfoque leve.
- Incluye varios objetos por imagen, objetos parcialmente tapados y algunas imágenes **sin ningún objeto** (fondos puros; `.txt` vacío o inexistente).
- Evita fotos casi idénticas (ráfagas): inflan el dataset sin aportar.
- Resolución mínima recomendada: 640×480. Se reducen a 256×256 al entrenar, así que **objetos muy pequeños (moneda lejana) se pierden**: acércate para que cada objeto ocupe al menos ~5 % del ancho de la imagen.
- Renombra los archivos sin espacios ni tildes (`moneda_001.jpg`).

## 3. Reglas de etiquetado (consistencia > cantidad)
1. **Etiqueta todos** los objetos de las 5 clases visibles; un objeto sin etiquetar se aprende como "fondo" y perjudica al modelo.
2. Caja **ajustada**: toca los bordes visibles del objeto, sin margen grande ni recortes.
3. Objetos parcialmente tapados: caja solo sobre la parte visible (si se ve ≥ ~30 %). Si casi no se ve, omítelo.
4. Objetos cortados por el borde de la imagen: caja hasta el borde.
5. Un objeto = una caja. No agrupes varias monedas en una sola caja.
6. Si dudas de la clase, descarta esa imagen en vez de adivinar.
7. Mantén el mismo criterio en todo el dataset (si etiquetas el bolígrafo con la tapa, siempre con la tapa).

## 4. Herramientas
### Opción A: LabelImg (local, simple)
```bash
pip install labelImg        # (puede requerir Python ≤3.10 y PyQt5)
labelImg data/train/images
```
1. Menú izquierdo: **cambia el formato a YOLO** (botón "PascalVOC" → "YOLO"). Es el paso que más se olvida.
2. *Change Save Dir* → `data/train/labels`.
3. Crea/edita `classes.txt` con las 5 clases **en el orden de `CLASES`**:
   ```
   moneda
   laptop
   boligrafo
   raton
   libro
   ```
4. Atajos: `W` crear caja, `D` siguiente imagen, `A` anterior, `Ctrl+S` guardar, `Del` borrar caja. Activa *Auto Save mode*.
5. Elimina el `classes.txt` que genera dentro de `labels/` si estorba (el proyecto ignora archivos que no sean `<imagen>.txt`, pero mantén una copia fuera).

### Opción B: Roboflow (web, con equipo)
1. Crea proyecto *Object Detection* y sube las imágenes.
2. Define las clases en el orden de `CLASES` y anota.
3. *Generate → Export Dataset → formato **YOLOv8*** (o "YOLO TXT"). **Desactiva** preprocesado y aumentos al generar; el proyecto ya aumenta solo.
4. Descomprime y copia `images/` y `labels/` a `data/train` y `data/val`.
5. Verifica los índices: Roboflow ordena las clases alfabéticamente por defecto, puede no coincidir con `CLASES`. Ver sección 6.

### Opción C: CVAT / Label Studio
Exporta como **YOLO 1.1** (CVAT) o YOLO (Label Studio) y revisa el orden de clases igual que arriba.

## 5. Dividir en train y val
Cuando termines de etiquetar, mueve ~20 % de las parejas imagen+txt a `val`:
```bash
cd data
mkdir -p train/images train/labels val/images val/labels
# mover cada 5.ª imagen (y su etiqueta) a val
i=0; for f in train/images/*; do
  i=$((i+1)); [ $((i % 5)) -ne 0 ] && continue
  b=$(basename "${f%.*}")
  mv "$f" val/images/; [ -f train/labels/$b.txt ] && mv train/labels/$b.txt val/labels/
done
```
Idealmente separa por escena o sesión de fotos para que val no tenga fotos casi iguales a train.

## 6. Verificar el dataset antes de entrenar
El repositorio incluye `verificar_dataset.py`, que detecta errores comunes:
```bash
python verificar_dataset.py
```
Lo que comprueba (resumen del código):
```python
from pathlib import Path
import config as C
for split in ("train", "val"):
    imgs = [p for p in (C.DATA_DIR/split/"images").iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")]
    conteo = [0]*C.NUM_CLASES; vacias = 0
    for p in imgs:
        txt = C.DATA_DIR/split/"labels"/(p.stem+".txt")
        lineas = txt.read_text().split("\n") if txt.exists() else []
        lineas = [l for l in lineas if l.strip()]
        vacias += not lineas
        for l in lineas:
            v = l.split()
            assert len(v) == 5, f"{txt}: línea mal formada: {l}"
            c = int(float(v[0])); assert 0 <= c < C.NUM_CLASES, f"{txt}: clase {c} inválida"
            assert all(0 <= float(x) <= 1 for x in v[1:]), f"{txt}: valores fuera de 0..1"
            conteo[c] += 1
    print(split, len(imgs), "imágenes,", vacias, "sin objetos |", dict(zip(C.CLASES, conteo)))
```
Ejecútalo con `python verificar_dataset.py`. Revisa que:
- Ninguna clase tenga mucho menos ejemplos que las demás (desbalance).
- No haya errores de rango o de índice de clase.
- Si Roboflow/CVAT reordenó clases, abre 5–10 etiquetas y comprueba visualmente que "0" es realmente una moneda.

## 7. Revisión visual (recomendado)
Abre varias imágenes en LabelImg con las etiquetas cargadas y comprueba que las cajas coinciden con los objetos y que las clases son correctas. Corregir 20 etiquetas malas rinde más que añadir 200 fotos.

## 8. Checklist final
- [ ] Cada imagen tiene su `.txt` (o es fondo intencional).
- [ ] Formato YOLO, valores entre 0 y 1, índices según `CLASES`.
- [ ] Cajas ajustadas y todos los objetos etiquetados.
- [ ] Clases balanceadas (≥ 300 por clase).
- [ ] train/val separados sin imágenes repetidas.
- [ ] `verificar_dataset.py` sin errores.
