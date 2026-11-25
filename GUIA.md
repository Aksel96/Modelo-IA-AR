# Guía paso a paso: de cero a un `.tflite`

## 1. Requisitos
- Python **3.10, 3.11 o 3.12** (TensorFlow no soporta 3.14).
- ~4 GB libres de disco. GPU opcional (con CPU funciona, solo es más lento).

## 2. Instalación
**Linux / macOS**
```bash
git clone https://github.com/<tu-usuario>/detectorIA.git
cd detectorIA
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**Windows (PowerShell)**
```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Comprueba: `python -c "import tensorflow as tf; print(tf.__version__)"`.

## 3. Estructura del dataset
```
data/
├── train/
│   ├── images/   foto001.jpg, foto002.jpg, ...
│   └── labels/   foto001.txt, foto002.txt, ...
└── val/
    ├── images/
    └── labels/
```
Reglas:
- Cada imagen tiene un `.txt` **con el mismo nombre** (`foto001.jpg` ↔ `foto001.txt`).
- Formatos de imagen: `.jpg .jpeg .png .bmp`.
- Cada línea del `.txt` es un objeto: `clase cx cy w h`
  - `clase`: índice según `CLASES` en `config.py` → `0` boligrafo, `1` laptop, `2` libro, `3` moneda, `4` mouse.
  - `cx cy w h`: centro y tamaño de la caja, **normalizados entre 0 y 1** (dividiendo por ancho/alto de la imagen).
  - Ejemplo (`foto001.txt`): 
    ```
    0 0.512 0.430 0.080 0.095
    4 0.250 0.700 0.300 0.220
    ```
- Una imagen sin objetos puede tener `.txt` vacío o no tenerlo (se usa como fondo).
- Máximo 20 objetos por imagen (`MAX_CAJAS` en `config.py`).

**Cómo repartir train/val:** ~80 % / 20 %, **sin repetir imágenes** entre ambos y, si puedes, separando por escena (que fotos casi idénticas no queden en los dos).

**Cuántos datos:** al entrenar desde cero, apunta a **≥ 300 imágenes por clase** (mejor 500+), con fondos, luces, distancias y ángulos variados, y objetos parciales o con oclusión.

**Cómo etiquetar:** guía detallada en [GUIA_ETIQUETADO.md](GUIA_ETIQUETADO.md). Herramientas: [LabelImg](https://github.com/HumanSignal/labelImg), [CVAT](https://www.cvat.ai/), [Roboflow](https://roboflow.com/) o [Label Studio](https://labelstud.io/) y exporta en **formato YOLO** (el orden de clases debe coincidir con `CLASES`).

## 4. Verificar el pipeline (opcional)
```bash
python generar_dataset_sintetico.py --n 60   # crea datos de juguete en data/
python train.py --epocas 2 --batch 8
```
Si funciona, **borra el contenido de `data/train` y `data/val`** (o las carpetas `images/` y `labels/`) antes de poner tus datos reales.

## 5. Entrenar
```bash
python train.py --epocas 150 --batch 16
```
- Guarda el mejor modelo en `salida/best_model.keras` (menor `val_loss`).
- Reduce el learning rate si la validación se estanca y se detiene sola si no mejora en 30 épocas (`PACIENCIA` en `config.py`, o `--paciencia`).
- **Orientación EXIF:** si tus fotos traen etiqueta EXIF de orientación (fotos de celular), entrena y evalúa con `APLICAR_EXIF=1` (ver README); las cajas deben dibujarse sobre la imagen ya orientada.
- Si te quedas sin memoria, baja `--batch` (8 o 4).

## 6. Evaluar
```bash
python evaluate.py --score 0.5
```
Muestra AP@0.5 por clase, Precision, Recall y mAP@0.5 sobre `val/`.
Si el mAP es bajo: más datos, revisa etiquetas, entrena más épocas.

## 7. Exportar a TFLite
```bash
python export_tflite.py --cuantizar
```
Genera `salida/detector.tflite` y `salida/labels.txt`.

## 8. Usarlo en la app de RA
- Entrada: `[1, 256, 256, 3]` float32, píxeles en **[0, 1]**.
- Salidas: `boxes [1, 2046, 4]` (ymin, xmin, ymax, xmax normalizado) y `scores [1, 2046, 6]` (índice 0 = fondo). Identifícalas por su shape.
- Por clase: descarta score < 0.5 y aplica NMS con IoU 0.45 (la app lo hace; no va dentro del modelo).
- `labels.txt`: la línea N corresponde al índice N de `scores` (línea 0 = "fondo").

## 9. Usar otras 5 categorías
Edita `CLASES` en `config.py`, reetiqueta con esos índices y repite desde el paso 5.

## Problemas frecuentes
| Síntoma | Causa / solución |
|---|---|
| `No module named tensorflow` | Entorno virtual no activado o Python 3.13+/3.14. |
| `FileNotFoundError ... images` | Falta `data/train/images` o `data/val/images` con imágenes. |
| `Clase N fuera de rango` | Un `.txt` usa un índice ≥ nº de clases. |
| `val_loss` no baja | Pocos datos o etiquetas mal normalizadas (valores >1). |
| Sin memoria | Reduce `--batch`. |
