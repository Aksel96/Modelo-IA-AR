# Bitácora de reentrenamiento: guía para tomar notas

Sirve para ir anotando en borrador lo que se hace y por qué. Después se pasa a Claude para que lo redacte en limpio como reporte.

**Cómo usarla**
1. Copia este archivo (por ejemplo a `BITACORA_BORRADOR.md`) y rellena las secciones mientras trabajan.
2. No hace falta redactar bien: sirven frases cortas, números y pegar salidas de consola. Lo importante es **anotar el dato en el momento**, con fecha.
3. Lo que no sepas, déjalo como `[PENDIENTE]`. No lo inventes.
4. Cuando tengas el borrador, pásalo completo. Claude lo ordena, une los números y redacta el texto.

> Regla: cada decisión se anota con su **razón** y su **resultado**. Un cambio sin resultado medido no sirve para el reporte.

---

## 1. Contexto y objetivo
- **Proyecto:** detector SSD-MobileNetV2 desde cero (`weights=None`), 5 clases: moneda, laptop, boligrafo, raton, libro. Se exporta a TFLite para una app de RA.
- **Objetivo de esta etapa:** `[ej.: lograr mAP@0.5 ≥ X en val]` `[PENDIENTE]`
- **Quién hizo qué:** `[PENDIENTE]`
- **Fechas (inicio / fin):** `[PENDIENTE]`

## 2. Punto de partida: primera ejecución (ya medida)
| Dato | Valor |
|---|---|
| Hardware | Windows, solo CPU |
| Épocas programadas / ejecutadas | 150 / 42 |
| Mejor época | 22 (`val_loss` = 9.2566) |
| `val_loss` en la época 42 | 10.5427 |
| Learning rate final | 1.25e-4 (bajó 3 veces desde 1e-3) |
| Causa de la parada | `EarlyStopping(patience=20)`: 22 + 20 = 42 |
| Tiempo por época | 1–5 min |
| mAP@0.5 | no se midió `[PENDIENTE: medirlo con el modelo archivado]` |

**Diagnóstico:** `val_loss` subió mientras el entrenamiento seguía. Es sobreajuste con pocos datos (100–300 imágenes por clase). El nivel de aprendizaje no fue el óptimo.

## 3. Entorno nuevo (WSL2 + GPU)
Anota lo que usan realmente:
- **GPU / VRAM:** `[PENDIENTE]` (comando: `nvidia-smi`)
- **Driver NVIDIA (Windows):** `[PENDIENTE]`
- **Distribución WSL:** `[PENDIENTE]`
- **Python / TensorFlow:** `[PENDIENTE]` (comando: `python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"`)
- **Ubicación del proyecto:** `~/Modelo-IA-AR` o `/mnt/c/...` `[PENDIENTE]`
- **Problemas de instalación y cómo se resolvieron:** `[PENDIENTE]`
- **Tiempo por época con GPU:** `[PENDIENTE]` (comparar con 1–5 min en CPU)
- **Uso de GPU durante el entrenamiento (`nvidia-smi`):** `[PENDIENTE]`

## 4. Aumento del dataset
Es la parte más importante del reporte. Anota antes de empezar, durante y al terminar.

### 4.1 Por qué se amplía
- El modelo no alcanzó un aprendizaje óptimo: `val_loss` subió desde la época 22.
- Se entrena desde cero, sin ImageNet; la guía recomienda ≥ 300 imágenes por clase y hay 100–300.
- `[añadan aquí los argumentos propios]`

### 4.2 Conteos antes y después
Se obtienen con `python verificar_dataset.py` (imágenes y cajas por clase). Pega la salida y rellena:

| Clase | Imágenes train (antes) | Imágenes train (después) | Cajas train (antes) | Cajas train (después) | Cajas val (antes) | Cajas val (después) |
|---|---|---|---|---|---|---|
| moneda | | | | | | |
| laptop | | | | | | |
| boligrafo | | | | | | |
| raton | | | | | | |
| libro | | | | | | |
| **Total imágenes** | | | | | | |

Anota también las imágenes sin objetos (fondos).

### 4.3 Cómo se consiguieron las imágenes nuevas
- **Fuente:** fotos propias, internet, capturas de video, otra fuente `[PENDIENTE]`
- **Dispositivo y resolución:** `[PENDIENTE]`
- **Quién tomó o recopiló cada lote y cuántas imágenes:** `[PENDIENTE]`
- **Variedad buscada:** fondos, luz, distancias, ángulos, objetos parciales u ocultos `[PENDIENTE]`
- **Licencias o permisos, si vienen de internet:** `[PENDIENTE]`

### 4.4 Etiquetado
- **Herramienta** (LabelImg, CVAT, Roboflow, Label Studio) y **formato** (YOLO): `[PENDIENTE]`
- **Orden de clases verificado** contra `CLASES` en `config.py`: sí / no
- **Criterios acordados** (cajas ajustadas, objetos parciales, qué hacer con oclusión): ver [GUIA_ETIQUETADO.md](GUIA_ETIQUETADO.md) `[PENDIENTE]`
- **Revisión de calidad:** quién revisó, qué porcentaje, errores encontrados `[PENDIENTE]`
- **Resultado de `verificar_dataset.py` tras añadir datos:** `Dataset OK` / errores `[PENDIENTE]`

### 4.5 Reparto train / val
- Proporción usada (objetivo ~80 / 20): `[PENDIENTE]`
- **Sin repetir imágenes entre train y val:** sí / no
- **Fotos casi idénticas separadas por escena:** sí / no
- Si val también creció o se dejó igual, y por qué `[PENDIENTE]`. Cambiarlo afecta la comparación con la primera ejecución.

### 4.6 Aumento de datos en código (opcional)
Hoy `data.py` solo aplica flip horizontal, brillo y contraste. Si se añade algo más (recortes, escala, rotación leve), anota qué, por qué y el resultado.

## 5. Configuración de cada entrenamiento
Una fila por ejecución. Pega el comando exacto.

| Ejecución | Fecha | Comando | Batch | LR | Patience | Dataset (imgs train/val) | Notas |
|---|---|---|---|---|---|---|---|
| Run 1 (CPU) | | `python train.py --epocas 150 --batch 16` | 16 | 1e-3 | 20 | `[PENDIENTE]` | referencia |
| Run 2 | | | | | | | |
| Run 3 | | | | | | | |

Cambios de código respecto a la primera ejecución (`train.py`): `--patience` configurable (por defecto 40), `salida/historial.csv` por época y aviso si TensorFlow no ve la GPU.

## 6. Resultados
Por ejecución, usando `salida/historial.csv` y `python evaluate.py --score 0.5`.

| Ejecución | Épocas ejecutadas | Mejor época | `val_loss` mejor | `loss` final | Tiempo total | mAP@0.5 | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| Run 1 (CPU) | 42 | 22 | 9.2566 | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| Run 2 | | | | | | | | |
| Run 3 | | | | | | | | |

**AP@0.5 por clase** (de `evaluate.py`):

| Clase | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| moneda | | | |
| laptop | | | |
| boligrafo | | | |
| raton | | | |
| libro | | | |

Si se pueden, adjunta la gráfica de `loss` y `val_loss` por época (sale de `historial.csv`).

## 7. Análisis
Responder en pocas líneas, con base en los números de arriba:
- ¿Mejoró el `val_loss` y el mAP con el dataset ampliado? ¿Cuánto?
- ¿Sigue habiendo sobreajuste? ¿En qué época empieza?
- ¿Qué clase rinde peor y por qué (pocos datos, se parece a otra, etiquetas dudosas)?
- ¿Cuánto aportó la GPU? Tiempo por época antes y después.
- ¿Subir la paciencia a 40 cambió algo?

## 8. Decisiones tomadas
| Fecha | Decisión | Razón | Resultado |
|---|---|---|---|
| | | | |

## 9. Problemas encontrados
| Problema | Síntoma / mensaje de error | Causa | Solución |
|---|---|---|---|
| | | | |

## 10. Pendientes y siguientes pasos
- [ ] Medir el mAP del modelo de la primera ejecución (para comparar con la misma métrica).
- [ ] Decidir si el resultado alcanza para exportar a TFLite (`python export_tflite.py --cuantizar`).
- [ ] Probar el `.tflite` en la app de RA.
- [ ] `[añadan los suyos]`

## 11. Anexos (adjuntar o pegar)
- Salida completa de `python verificar_dataset.py` (antes y después).
- `salida/historial.csv` de cada ejecución.
- Salida de `python evaluate.py` de cada modelo.
- Captura de `nvidia-smi` con el entrenamiento en marcha.
- Resultado de `pip freeze` (versiones exactas).

---

## Comandos para sacar los datos rápido
```bash
python verificar_dataset.py                       # conteos por clase, train y val
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
nvidia-smi                                        # GPU, VRAM y driver
pip freeze > requisitos_exactos.txt               # versiones exactas
python evaluate.py --score 0.5                    # mAP, Precision, Recall
```

Resumen de `historial.csv` (mejor época y valores finales):
```bash
python - <<'EOF'
import csv
f = list(csv.DictReader(open("salida/historial.csv")))
m = min(f, key=lambda r: float(r["val_loss"]))
print("épocas:", len(f), "| mejor época:", int(m["epoch"]) + 1, "| val_loss:", m["val_loss"])
print("loss final:", f[-1]["loss"], "| val_loss final:", f[-1]["val_loss"])
EOF
```

## Qué pasarle a Claude para el documento final
1. Este borrador con las secciones rellenas.
2. Las salidas de los comandos de arriba (pegadas o como archivos).
3. A quién va dirigido el reporte (profesor, equipo, cliente) y la extensión que quieren.
