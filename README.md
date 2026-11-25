# Detector de objetos SSD-MobileNetV2 **sin pesos preentrenados**

Detecta 5 clases (`boligrafo, laptop, libro, moneda, mouse`, orden de `CLASES` en `config.py`), se entrena desde cero
(`weights=None`, nunca se descarga ImageNet) y se exporta a TFLite para una app de RA.

> 📘 Guía completa paso a paso (instalación, estructura del dataset, entrenamiento y uso del `.tflite`): [GUIA.md](GUIA.md)

## Instalación
Usa Python 3.10–3.12 (TensorFlow aún puede no soportar versiones más nuevas).
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset (formato YOLO)
```
data/train/images/*.jpg   data/train/labels/*.txt
data/val/images/*.jpg     data/val/labels/*.txt
```
Cada línea del `.txt`: `clase cx cy w h` (normalizados 0–1; clase = índice en `CLASES`).
Una imagen sin `.txt` cuenta como solo fondo.

Prueba rápida del pipeline con datos sintéticos (no sirve para un detector real):
```bash
python generar_dataset_sintetico.py --n 60
```

## Ejecución
```bash
python train.py --epocas 150           # guarda salida/best_model.keras
python evaluate.py                     # Precision, Recall, mAP@0.5 en val/
python export_tflite.py --cuantizar    # salida/detector.tflite + salida/labels.txt
```

## Orientación EXIF (bandera `APLICAR_EXIF`)
**Orientación EXIF.** Las etiquetas están dibujadas sobre la imagen *orientada* (como se ve en cualquier visor), pero `tf.io.decode_image` devuelve los píxeles sin rotar. Con `APLICAR_EXIF=1` el pipeline aplica la orientación EXIF antes del resize (modo por defecto = legacy, sin rotar). Úsala **igual al entrenar y al evaluar**: evaluar con la bandera desactivada un modelo entrenado con ella da resultados no válidos.
```bash
APLICAR_EXIF=1 python train.py --epocas 150
APLICAR_EXIF=1 python evaluate.py --score 0.5
```
Tests: `python -B -m unittest tests.test_exif -v` (no entrenan).

## Archivos
| Archivo | Función |
|---|---|
| `config.py` | Clases, rutas e hiperparámetros |
| `anchors.py` | Anchors SSD, codificación/decodificación, IoU |
| `data.py` | Lectura YOLO, orientación EXIF opcional, `tf.data`, aumento (flip, zoom, brillo, contraste), asignación de anchors |
| `model.py` | SSD con `MobileNetV2(weights=None)` + cabezas de clasificación/regresión |
| `losses.py` | Smooth L1 + `SparseCategoricalCrossentropy` con hard negative mining 3:1 |
| `train.py` | `model.fit`, Adam, `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping` |
| `evaluate.py` | NMS, Precision, Recall, mAP@0.5 (IoU propio) |
| `export_tflite.py` | Conversión TFLite (cuantización dinámica opcional) y `labels.txt` |

## Salida del TFLite
- **Entrada**: `[1, 256, 256, 3]` float32 con valores en **[0, 1]** (la normalización va dentro del modelo).
- **Salidas** (identifícalas por su shape, el orden de los tensores en TFLite puede variar): `boxes [1, 2046, 4]` = (ymin, xmin, ymax, xmax) normalizado y `scores [1, 2046, 6]`
  (probabilidades; índice 0 = fondo, 1..5 = clases en el orden de `labels.txt`, que ya incluye "fondo" en la línea 0).
- **NMS**: se aplica en la app (por clase: filtrar score ≥ 0.5, IoU ≤ 0.45), no va dentro del grafo.

## Adaptar a otras 5 categorías
Edita solo `CLASES` en `config.py` y reetiqueta con esos índices. Todo lo demás se ajusta solo.

## Consejos (entrenar desde cero)
Sin preentrenamiento hacen falta muchos más datos y épocas: apunta a ≥ 300 imágenes por clase,
con variedad de fondos, luz y ángulos, y entrena 100+ épocas. Si val_loss no baja, revisa primero
que las etiquetas estén bien (clase y coordenadas normalizadas).

## Estado de verificación
Probado de extremo a extremo con el dataset sintético (Python 3.10, TensorFlow 2.21): entrenamiento,
evaluación, exportación TFLite cuantizado y carga con `tf.lite.Interpreter` funcionan. Con solo 2 épocas
el mAP es ~0, lo esperado; la calidad real depende de tu dataset.
