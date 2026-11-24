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
- **Proyecto:** detector SSD-MobileNetV2 desde cero (`weights=None`), 5 clases: `["boligrafo", "laptop", "libro", "moneda", "mouse"]`. Se exporta a TFLite para una app de RA.
- **Objetivo de esta etapa:** Establecer la línea base (Run 1), habilitar GPU en WSL2, rebalancear el dataset inicial (Run 2: mAP subió de 0.000 a 0.065), y preparar la ampliación a 1,410 imágenes para superar el sobreajuste y alcanzar un mAP viable ($\ge 0.50$).
- **Quién hizo qué:** Equipo de desarrollo e investigación en IA.
- **Fechas (inicio / fin):** Noviembre 2026.

## 2. Punto de partida: primera ejecución (Run 1 - Baseline CPU)
| Dato | Valor |
|---|---|
| Hardware | Windows 11 nativo, solo CPU (TF 2.21 sin GPU en Windows) |
| Épocas programadas / ejecutadas | 150 / 42 |
| Mejor época | 22 (`val_loss` = 9.2566, `loss` = 2.6298) |
| `val_loss` en la época 42 | 10.5427 (`loss` en época 42 = 0.7266) |
| Learning rate final | 6.25e-5 (reducido por `ReduceLROnPlateau` en épocas 27, 32, 37 y 42) |
| Causa de la parada | `EarlyStopping(patience=20)`: 22 + 20 = 42 épocas consecutivas sin mejora |
| Tiempo por época | ~19 a 20 s por época (~14 minutos en total) |
| mAP@0.5 | 0.000 (Precision: 0.000, Recall: 0.000, todas las clases en 0.000 con score $\ge 0.5$) |

**Diagnóstico:** `val_loss` se estancó en 9.2566 y comenzó a subir hacia 10.5427 mientras `loss` bajaba a 0.7266. Esto evidenció un sobreajuste (*overfitting*) severo provocado por la escasez de datos (solo 50 imágenes por clase en `train`, 250 en train frente a 205 en val; relación 55% train / 45% val). El modelo memorizó el set de entrenamiento sin capacidad de generalización.

## 3. Entorno nuevo (WSL2 + GPU)
- **GPU / VRAM:** NVIDIA GeForce RTX 4060 Laptop/Desktop (8,188 MiB VRAM)
- **Driver NVIDIA (Windows):** 621.01 (CUDA UMD Version 13.5)
- **Distribución WSL:** Ubuntu (WSL2)
- **Python / TensorFlow:** TensorFlow con `tensorflow[and-cuda]` (detectó exitosamente `/physical_device:GPU:0`)
- **Ubicación del proyecto:** `/mnt/d/Respaldo/Documents/Escuela/InteligenciaArtificial/Proyecto/Modelo-IA-AR`
- **Problemas de instalación y cómo se resolvieron:** TensorFlow $\ge 2.11$ retiró el soporte nativo GPU en Windows (`WARNING:tensorflow:TensorFlow GPU support is not available on native Windows`). Se resolvió activando WSL2 e instalando el stack de Linux con soporte CUDA directo.
- **Tiempo por época con GPU:** ~1 a 2 segundos por época (aceleración de $10\times$ a $15\times$ frente a CPU).
- **Uso de GPU durante el entrenamiento (`nvidia-smi`):** Activo, memoria asignada para cálculo de tensores y gradientes.

## 4. Aumento del dataset
Es la parte más importante del reporte. Anota antes de empezar, durante y al terminar.

### 4.1 Por qué se amplía
- El modelo no alcanzó un aprendizaje óptimo: `val_loss` subió desde la época 22.
- Se entrena desde cero, sin ImageNet; la guía recomienda ≥ 300 imágenes por clase y hay 100–300.
- `[añadan aquí los argumentos propios]`

### 4.2 Conteos antes y después
Se obtienen con `python verificar_dataset.py` (imágenes y cajas por clase).

| Clase | Cajas train (Run 1 Inicial) | Cajas val (Run 1 Inicial) | Cajas val (Run 2 Rebalanceo) | Cajas train (Fase 2 - 1410 imgs) | Cajas val (Fase 2) |
|---|---|---|---|---|---|
| boligrafo | 50 | 41 | 19 | `[PENDIENTE]` | `[PENDIENTE]` |
| laptop | 50 | 48 | 26 | `[PENDIENTE]` | `[PENDIENTE]` |
| libro | 50 | 45 | 23 | `[PENDIENTE]` | `[PENDIENTE]` |
| moneda | 50 | 30 | 8 | `[PENDIENTE]` | `[PENDIENTE]` |
| mouse | 50 | 41 | 19 | `[PENDIENTE]` | `[PENDIENTE]` |
| **Total imágenes** | **250** | **205** | **~95 GTs** | **~1,097 (77.8%)** | **~313 (22.2%)** |

> *Nota:* En la Fase inicial se tenían 455 imágenes totales. En la nueva fase se vació la data previa para cargar **1,410 imágenes nuevas** con mayor densidad de objetos por escena.

### 4.3 Cómo se consiguieron las imágenes nuevas
- **Fuente:** Recolección propia mediante capturas fotográficas en ráfagas / secuencias continuas.
- **Total de imágenes recolectadas:** 1,410 imágenes.
- **Variedad buscada:** Distintos ángulos, distancias, condiciones de luz y fondos cotidianos para evitar memorización de entorno.

### 4.4 Etiquetado
- **Herramienta y formato:** Formato YOLO (`clase cx cy w h` normalizado 0..1).
- **Orden de clases verificado:** Sí, coincidente con `CLASES` en `config.py` (`0: boligrafo, 1: laptop, 2: libro, 3: moneda, 4: mouse`).
- **Criterios acordados:** Cajas ajustadas, objetos parcialmente visibles etiquetados si superan ~30% de visibilidad.

### 4.5 Reparto train / val
- **Fase 1 (Inicial):** 250 train / 205 val (reparto deficiente de 55% / 45%). Rebalanceado luego a ~360 train / ~95 val.
- **Fase 2 (1,410 imágenes):** Reparto sistemático mediante script con patrón alternado de paso (3 train $\rightarrow$ 1 val, 4 train $\rightarrow$ 1 val).
  - Ciclo de 9 fotos consecutivas: 7 a `train` (77.8%) y 2 a `val` (22.2%).
  - Objetivo: Desacoplar fotos consecutivas para evitar filtración de datos (*data leakage*) entre entrenamiento y validación, manteniendo representatividad temporal.

### 4.6 Aumento de datos en código (opcional)
Hoy `data.py` solo aplica flip horizontal, brillo y contraste. Si se añade algo más (recortes, escala, rotación leve), anota qué, por qué y el resultado.

## 5. Configuración de cada entrenamiento
Una fila por ejecución. Pega el comando exacto.

| Ejecución | Fecha | Comando | Batch | LR | Patience | Dataset (imgs train/val) | Notas |
|---|---|---|---|---|---|---|---|
| Run 1 (CPU) | Nov 2026 | `python train.py --epocas 150 --batch 16` | 16 | 1e-3 | 20 | 250 / 205 | Baseline en Windows CPU. Overfitting severo. mAP: 0.000 |
| Run 2 (WSL2 GPU) | Nov 2026 | `python3 train.py --paciencia 30` | 16 | 1e-3 | 30 | ~360 / ~95 | WSL2 + RTX 4060. Rebalanceo inicial. mAP subió a 0.065 |
| Run 3 (Nueva Data) | `[PENDIENTE]` | `python3 train.py --paciencia 30` | 16 | 1e-3 | 30 | ~1097 / ~313 (1410 total) | Prueba con dataset ampliado |

## 6. Resultados
Por ejecución, usando `salida/historial.csv` y `python evaluate.py --score 0.5`.

| Ejecución | Épocas ejecutadas | Mejor época | `val_loss` mejor | `loss` final | Tiempo por época | mAP@0.5 | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| Run 1 (CPU) | 42 | 22 | 9.2566 | 0.7266 | ~20 s | 0.000 | 0.000 | 0.000 |
| Run 2 (WSL2) | ~50+ | - | - | - | ~1-2 s | 0.065 | 0.000 (score $\ge 0.5$) | 0.000 (score $\ge 0.5$) |
| Run 3 | `[PENDIENTE]` | | | | | | | |

**AP@0.5 por clase** (de `evaluate.py`):

| Clase | Run 1 (Baseline) | Run 2 (Rebalanceo previo) | Run 3 (1,410 imágenes) |
|---|---|---|---|
| boligrafo | 0.000 (GT=41) | 0.009 (GT=19) | `[PENDIENTE]` |
| laptop | 0.000 (GT=48) | 0.082 (GT=26) | `[PENDIENTE]` |
| libro | 0.000 (GT=45) | 0.002 (GT=23) | `[PENDIENTE]` |
| moneda | 0.000 (GT=30) | 0.000 (GT=8) | `[PENDIENTE]` |
| mouse | 0.000 (GT=41) | 0.234 (GT=19) | `[PENDIENTE]` |
| **mAP@0.5 General** | **0.000** | **0.065** | `[PENDIENTE]` |

## 7. Análisis
- **Evolución Run 1 $\rightarrow$ Run 2:** Con solo 250 imágenes en train (Run 1), la red MobileNetV2 de 2.6M de parámetros sufrió sobreajuste absoluto (mAP=0.000). Al rebalancear el split a favor de train e incrementar la paciencia a 30 épocas (Run 2), el mAP subió a 0.065, demostrando que `mouse` (AP 0.234) y `laptop` (AP 0.082) ya aprendían patrones representativos.
- **Impacto de la GPU:** La migración a WSL2 con la RTX 4060 aceleró el entrenamiento de ~20 segundos/época a ~1-2 segundos/época (reducción de tiempo superior al 90%).
- **Motivación para Run 3:** 455 imágenes siguen siendo insuficientes para un entrenamiento desde cero (`weights=None`). Se recolectaron 1,410 imágenes para dotar al modelo de la variabilidad estadística necesaria para superar un mAP de 0.40–0.60.

## 8. Decisiones tomadas
| Fecha | Decisión | Razón | Resultado |
|---|---|---|---|
| Nov 2026 | Migración de entrenamiento a WSL2 (Ubuntu) con `tensorflow[and-cuda]` | TensorFlow $\ge 2.11$ no tiene aceleración GPU en Windows nativo | Detección exitosa de NVIDIA RTX 4060; épocas bajan de 20s a 1-2s |
| Nov 2026 | Actualización de orden de `CLASES` a orden alfabético | Estandarización de clases (`boligrafo, laptop, libro, moneda, mouse`) | Mapeo ordenado en `config.py` y salida en `labels.txt` |
| Nov 2026 | Ampliación de tolerancia de `EarlyStopping` de 20 a 30 épocas y parámetro `--paciencia` | Dar margen a que el modelo explore mejores mínimos locales antes de detenerse | Mayor estabilidad y convergencia en entrenamientos |
| Nov 2026 | Rebalanceo inicial de datos (55/45 $\rightarrow$ 80/20) | Evitar el desperdicio de datos de entrenamiento en validación | mAP subió de 0.000 a 0.065 |
| Nov 2026 | Recolección de 1,410 imágenes nuevas y separación intercalada (3-1, 4-1) | Escalar el dataset y evitar fuga de información por fotos consecutivas | En proceso para Run 3 |

## 9. Problemas encontrados
| Problema | Síntoma / mensaje de error | Causa | Solución |
|---|---|---|---|
| Falta de aceleración GPU | `WARNING:tensorflow:TensorFlow GPU support is not available on native Windows` | Limitación oficial de Google para TensorFlow en Windows desde la versión 2.11 | Se configuró entorno WSL2 Ubuntu con driver NVIDIA de Windows |
| Detención temprana en época 42 | Entrenamiento se detuvo mucho antes de las 150 épocas | `val_loss` no mejoró durante 20 épocas consecutivas tras la época 22 (sobreajuste) | Se aumentó la paciencia a 30 en `train.py` y se planificó aumento masivo de datos |
| Métricas en cero (mAP 0.000) | `Precision: 0.000`, `Recall: 0.000`, `AP: 0.000` en todas las clases | Muy pocos datos (50 fotos por clase) para entrenar MobileNetV2 sin ImageNet | Rebalanceo inicial (llevó mAP a 0.065) y recolección de 1,410 imágenes |

## 10. Pendientes y siguientes pasos
- [x] Documentar línea base (Run 1) y primer rebalanceo (Run 2) en la bitácora.
- [ ] Ejecutar script de partición intercalada para distribuir las 1,410 imágenes entre `data/train` y `data/val`.
- [ ] Verificar consistencia de etiquetas y clases con `python verificar_dataset.py`.
- [ ] Entrenar Run 3 en WSL2 con la RTX 4060 (`python3 train.py --paciencia 30`).
- [ ] Evaluar métricas de Run 3 con `python evaluate.py --score 0.5` y comparar evolución.
- [ ] Exportar a TFLite (`python export_tflite.py --cuantizar`) y validar en la app de RA.

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
