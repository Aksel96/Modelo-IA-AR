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

| Clase | Cajas train (Run 1) | Cajas val (Run 1) | Cajas val (Run 2) | Cajas train (Run 3 - Nueva Data) | Cajas val (Run 3) |
|---|---|---|---|---|---|
| boligrafo | 50 | 41 | 19 | 847 | 250 |
| laptop | 50 | 48 | 26 | 205 | 58 |
| libro | 50 | 45 | 23 | 308 | 88 |
| moneda | 50 | 30 | 8 | 602 | 168 |
| mouse | 50 | 41 | 19 | 212 | 60 |
| **Total imágenes** | **250** | **205** | **~95 GTs** | **1,097 (77.8%)** | **313 (22.2%)** |

> *Nota:* Reparto realizado con `repartir_recursos.py` aplicando el patrón sistemático 3-1, 4-1 sobre 1,410 imágenes consecutivas. Total cajas anotadas: 2,174 en train y 624 en val.

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

### 4.6 Aumento de datos en código
- **Flip horizontal:** 50% de probabilidad con inversión correspondiente del centro $c_x$ de las cajas.
- **Brillo y contraste aleatorios:** Variación de brillo ($\pm 0.2$) y contraste ($0.8 - 1.2$).
- **Random Zoom / Recorte aleatorio (implementado en Run 4):**
  - Función `_random_zoom(img, gt, p=0.5)` con factor de recorte $s \in [0.60, 0.90]$ (zoom equivalente de $1.1\times$ a $1.67\times$).
  - Centra el recorte alrededor de objetos válidos y reescala automáticamente las coordenadas de las cajas `(cx, cy, w, h)` al marco de $256 \times 256$.
  - Diseñado para forzar a que objetos pequeños como monedas y bolígrafos aumenten su área visible relativa entre un 30% y 60% durante el entrenamiento.

## 5. Configuración de cada entrenamiento
Una fila por ejecución. Pega el comando exacto.

| Ejecución | Fecha | Comando | Batch | LR | Patience | Dataset (imgs train/val) | Notas |
|---|---|---|---|---|---|---|---|
| Run 1 (CPU) | Nov 2026 | `python train.py --epocas 150 --batch 16` | 16 | 1e-3 | 20 | 250 / 205 | Baseline en Windows CPU. Overfitting severo. mAP: 0.000 |
| Run 2 (WSL2 GPU) | Nov 2026 | `python3 train.py --paciencia 30` | 16 | 1e-3 | 30 | ~360 / ~95 | WSL2 + RTX 4060. Rebalanceo inicial. mAP subió a 0.065 |
| Run 3 (1,410 imgs) | Nov 2026 | `python3 train.py --paciencia 30` | 16 | 1e-3 | 30 | 1,097 / 313 (1,410 total) | Dataset ampliado. Mejor val_loss 8.1028. mAP 0.061. Modelo archivado como best_m4.keras |
| Run 4 (Anchors 6 + Zoom) | Nov 2026 | `python3 train.py --epocas 150 --paciencia 30` | 16 | 1e-3 | 30 | 1,097 / 313 | 6 anchors/celda (2,046 total, ratios 3:1 y 1:3), escala min 0.07 y Random Zoom |
| Run 5 (E1: EXIF) | 24-nov-2026 | `APLICAR_EXIF=1 python train.py --epocas 150` (evaluación: `APLICAR_EXIF=1 python evaluate.py --score 0.5`) | 16 | 1e-3 | 30 (`config.PACIENCIA`; `ReduceLROnPlateau` patience 5) | 1,097 / 313 | **Único cambio respecto a Run 4: orientación EXIF aplicada** al cargar (commit `dd21daa`, rama `fix/e1-exif`). Mismos anchors, zoom e hiperparámetros. Lo ejecutó el usuario en WSL2. Archivos: `archivo/best_m5.keras`, `archivo/historial_m5.csv`, `archivo/evaluate_m5.txt` (ver §13.2) |

## 6. Resultados
Por ejecución, usando `salida/historial.csv` y `python evaluate.py`.

| Ejecución | Épocas ejecutadas | Mejor época | `val_loss` mejor | `loss` final | Tiempo por época | mAP@0.5 | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| Run 1 (CPU) | 42 | 22 | 9.2566 | 0.7266 | ~20 s | 0.000 | 0.000 | 0.000 |
| Run 2 (WSL2) | ~50+ | - | - | - | ~1-2 s | 0.065 | 0.000 (score $\ge 0.5$) | 0.000 (score $\ge 0.5$) |
| Run 3 (1,410 imgs) | 39 | ~9 | 8.1028 | 2.2294 | ~9 s | 0.061 | 0.000 (score $\ge 0.5$) | 0.000 (score $\ge 0.5$) |
| Run 4 (Anchors 6 + Zoom) | 41 | 11 | **7.5415** (Récord) | 3.3623 | ~10-12 s | 0.055 (score 0.5)<br>0.058 (score 0.2) | 0.000 (score 0.5)<br>0.003 (score 0.2) | 0.000 (score 0.5)<br>0.083 (score 0.2) |
| Run 5 (E1: EXIF) | 38 | 8 | 7.9220 | 2.5638 | ~12-14 s (época 1: 97 s por compilación XLA) | 0.055 (score 0.5; no se evaluó con 0.2) | 0.000 (score 0.5) | 0.000 (score 0.5) |

> Run 5: datos de `archivo/historial_m5.csv` (38 filas) y `archivo/evaluate_m5.txt`; detalle época por época en §13.2. **(Ver §14.9: con las estadísticas de BN recalibradas el mAP de este checkpoint es 0.283, no 0.055.)** **Su `val_loss` no es comparable con la de los Runs 1–4**: con EXIF cambia el marco de entrada de entrenamiento y validación (moneda y laptop). Solo el mAP y el AP por clase son comparables.

**AP@0.5 por clase** (de `evaluate.py`):

| Clase | Run 1 (Baseline) | Run 2 (Rebalanceo) | Run 3 (1,410 imgs) | Run 4 (Score 0.5) | Run 4 (Score 0.2) | Run 5 EXIF (Score 0.5) | Diagnóstico Geométrico / Dataset |
|---|---|---|---|---|---|---|---|
| boligrafo | 0.000 (GT=41) | 0.009 (GT=19) | 0.000 (GT=250) | **0.001** | **0.001** | 0.001 | Cajas diagonales capturan 95% de mesa; genera falsos positivos |
| laptop | 0.000 (GT=48) | 0.082 (GT=26) | 0.145 (GT=58) | **0.159** | **0.183** | 0.163 | **Mejora sólida**: objeto grande ($\approx 52\%$ área), excelente localización |
| libro | 0.000 (GT=45) | 0.002 (GT=23) | 0.145 (GT=88) | **0.110** | **0.104** | 0.109 | Estable ($\approx 40\%$ área), buena correspondencia geométrica |
| moneda | 0.000 (GT=30) | 0.000 (GT=8) | 0.000 (GT=168) | **0.000** | **0.000** | 0.000 | Escala microscópica al reducir $2048\times 1536 \rightarrow 256\times 256$ ($<12$ px) |
| mouse | 0.000 (GT=41) | 0.234 (GT=19) | 0.012 (GT=60) | **0.003** | **0.003** | 0.003 | Confusión con esquinas de laptops/libros y falta de tomas aisladas |
| **mAP@0.5 General** | **0.000** | **0.065** | **0.061** | **0.055** | **0.058** | **0.055** | **Mejor val_loss (7.5415)**, pero monedas y bolígrafos arrastran el promedio |

## 7. Análisis
- **Evolución de Pérdidas (Run 1 $\rightarrow$ Run 4):**
  - Run 1 (CPU): mejor `val_loss` = 9.2566.
  - Run 3 (1,410 imágenes): mejor `val_loss` = 8.1028.
  - Run 4 (Anchors 6 + Zoom): mejor `val_loss` = **7.5415** (en la época 11).
  - La convergencia matemática de la red continúa mejorando consistentemente con cada iteración.
- **Diagnóstico Profundo: El Dataset como Cuello de Botella:**
  1. **Efecto de Reducción de Resolución ($2048 \times 1536 \rightarrow 256 \times 256$ en Monedas):**
     - Las fotos originales se capturan a 3 MP. Al redimensionarse a $256 \times 256$ para MobileNetV2, sufren una reducción lineal de $8\times$ en cada eje (pérdida del 98.4% de los píxeles).
     - Una moneda tomada a distancia normal pasa de $\approx 80$ píxeles a tan solo **$8 \times 8$ o $12 \times 12$ píxeles**.
     - En la arquitectura SSD, la cuadrícula más fina (stride 16) tiene celdas de $16 \times 16$ píxeles. La moneda es más pequeña que una sola celda convolucional. Al entrenar **desde cero** (`weights=None`), la red no puede extraer bordes ni texturas de un objeto de 10 píxeles.
     > **Nota de corrección (24-nov-2026, ver §12):** esta explicación no se sostiene con los datos medidos. Las imágenes de moneda no son todas de 2048×1536 (hay 4032×3024) y las cajas etiquetadas, en el marco de la imagen orientada, tienen una mediana de ≈ 60 px a 256×256 (ninguna con área < 1 % de la imagen). Se identificó además una causa más directa: el desajuste EXIF (§12), que afectaba al 63–64 % de las cajas de moneda. La causa real del AP ≈ 0 de moneda sigue sin demostrarse hasta reentrenar con la corrección.
  2. **Contaminación de Fondo por Cajas Diagonales en Bolígrafos:**
     - En el dataset se detectaron cajas de bolígrafos muy anchas ($w=0.82, h=0.51$ o $w=0.94, h=0.53$).
     - Al fotografiar bolígrafos inclinados a 45°, la caja rectangular de YOLO abarca un 5% de bolígrafo real y un **95% de superficie plana (mesa, mantel o escritorio)**.
     - La red aprende que la textura de la mesa es un bolígrafo. Esto explica por qué a `score >= 0.2` la precisión cayó a **0.003**: la red detecta decenas de "bolígrafos fantasma" en cualquier espacio vacío de la mesa.
     > **Nota de corrección (24-nov-2026, ver §12):** en las 4 imágenes de bolígrafo inspeccionadas las cajas se ajustan razonablemente (con algo de holgura); los solapes entre cajas son físicos (bolígrafos amontonados en diagonal). El AP ≈ 0 de bolígrafo, que no tiene problema EXIF, apunta más bien a la cobertura de anchors (37 % de sus GT sin ningún anchor IoU ≥ 0.5; §12.6), hipótesis aún sin probar.
  3. **Sesgo Mono-clase y Penalización de Objetos no Anotados:**
     - Las imágenes del dataset están separadas estrictamente por clase (`laptop_xxx`, `mouse_xxx`, etc.).
     - Si en una foto de laptop aparece un mouse o un bolígrafo en el escritorio y **no fue anotado**, el algoritmo SSD trata esos anchors automáticamente como **FONDO**. Cada vez que la red reconoce el mouse en esa imagen, la pérdida la castiga activamente, destruyendo el aprendizaje de las clases secundarias.
     > **Nota de corrección (24-nov-2026, ver §12):** hipótesis no confirmada. Cada imagen contiene objetos de una sola clase y en unas 70 vistas inspeccionadas no se vieron objetos visibles sin anotar; no es posible detectarlos automáticamente.
  4. **Dinámica del Learning Rate (`ReduceLROnPlateau`):**
     - Tras alcanzar el récord de `val_loss: 7.5415` en la época 11, el callback con `patience=5` redujo el LR de forma demasiado agresiva ($1\times 10^{-3} \rightarrow 5\times 10^{-4} \rightarrow 2.5\times 10^{-4} \rightarrow 1.25\times 10^{-4} \rightarrow 6.25\times 10^{-5} \rightarrow 3.125\times 10^{-5} \rightarrow 1.56\times 10^{-5}$).
     - Al quedar con un LR tan diminuto en la época 26–31, la red quedó prácticamente congelada e incapaz de escapar de mesetas locales.

- **Run 5 (E1, EXIF activado; 24-nov-2026):** mAP@0.5 = 0.055, igual que Run 4; AP por clase prácticamente igual (laptop 0.163, libro 0.109, moneda 0.000, bolígrafo 0.001, mouse 0.003); precisión y recall 0.000 con score ≥ 0.5. Mejor `val_loss` 7.9220 en la época 8 (no comparable con Runs 1–4, ver nota en §6). Detalle y lectura en §13.2–13.3.

## 8. Decisiones tomadas
| Fecha | Decisión | Razón | Resultado |
|---|---|---|---|
| Nov 2026 | Migración de entrenamiento a WSL2 (Ubuntu) con `tensorflow[and-cuda]` | TensorFlow $\ge 2.11$ no tiene aceleración GPU en Windows nativo | Detección exitosa de NVIDIA RTX 4060; épocas bajan de 20s a 1-2s |
| Nov 2026 | Actualización de orden de `CLASES` a orden alfabético | Estandarización de clases (`boligrafo, laptop, libro, moneda, mouse`) | Mapeo ordenado en `config.py` y salida en `labels.txt` |
| Nov 2026 | Ampliación de tolerancia de `EarlyStopping` de 20 a 30 épocas y parámetro `--paciencia` | Dar margen a que el modelo explore mejores mínimos locales antes de detenerse | Mayor estabilidad y convergencia en entrenamientos |
| Nov 2026 | Rebalanceo inicial de datos (55/45 $\rightarrow$ 80/20) | Evitar el desperdicio de datos de entrenamiento en validación | mAP subió de 0.000 a 0.065 |
| Nov 2026 | Recolección de 1,410 imágenes nuevas y separación intercalada (3-1, 4-1) con `repartir_recursos.py` | Escalar el dataset y evitar fuga de información por fotos consecutivas | val_loss bajó a 8.1028; laptop y libro alcanzaron 0.145 |
| Nov 2026 | Archivado de Run 3 como `best_m4.keras` | Preservar el modelo de 1,410 imágenes antes de cambiar arquitectura de anchors | Respaldo seguro en `salida/best_m4.keras` *(corrección 24-nov-2026: el archivo está en `archivo/best_m4.keras`; `salida/` solo contiene `best_model.keras`)* |
| Nov 2026 | Ampliación a 6 anchors por celda (ratios 3:1 y 1:3), escala min 0.07 y Zoom/Crop en `data.py` | Solucionar las fallas de anclaje de bolígrafos y escala reducida de monedas | 2,046 anchors totales. val_loss cayó al mínimo histórico de 7.5415 |
| Nov 2026 | Modificación de `.gitignore` para incluir `data/` completo | Permitir subir el dataset (imágenes y etiquetas) al repositorio para auditoría y análisis | Dataset versionado y listo para inspección |
| 24-nov-2026 | E1: aplicar la orientación EXIF en el pipeline (propuesta A), bajo la bandera `APLICAR_EXIF` | Las etiquetas están en el marco de la imagen orientada y `decode_image` entrega píxeles sin rotar (§12) | Implementado y validado técnicamente (commit `dd21daa`). **Run 5: sin efecto en el mAP (0.055 → 0.055)**; la corrección es válida, pero no era la causa principal del AP ≈ 0 de moneda (§13.2) |
| 24-nov-2026 | Archivado de Run 5 como `archivo/best_m5.keras` + `archivo/historial_m5.csv` + `archivo/evaluate_m5.txt` | Conservar los datos reproducibles de cada run (regla 9.4 de CLAUDE.md); el `best_m5.keras` anterior (copia de Run 4) se sobrescribió por indicación del usuario y sigue recuperable en el commit `794a049` | Ver §13.4 |

## 9. Problemas encontrados
| Problema | Síntoma / mensaje de error | Causa | Solución |
|---|---|---|---|
| Falta de aceleración GPU | `WARNING:tensorflow:TensorFlow GPU support is not available on native Windows` | Limitación oficial de Google para TensorFlow en Windows desde la versión 2.11 | Se configuró entorno WSL2 Ubuntu con driver NVIDIA de Windows |
| Detención temprana en época 42 | Entrenamiento se detuvo mucho antes de las 150 épocas | `val_loss` no mejoró durante 20 épocas consecutivas tras la época 22 (sobreajuste) | Se aumentó la paciencia a 30 en `train.py` y se planificó aumento masivo de datos |
| Métricas en cero (mAP 0.000) | `Precision: 0.000`, `Recall: 0.000`, `AP: 0.000` en todas las clases | Muy pocos datos (50 fotos por clase) para entrenar MobileNetV2 sin ImageNet | Rebalanceo inicial (llevó mAP a 0.065) y recolección de 1,410 imágenes |
| Moneda indetectable (AP 0.000) | 0 aciertos en 168 objetos de validación | *(Causa original anotada: reducción de escala, moneda $<12$ px. **Corrección 24-nov-2026:** no respaldada por los datos medidos; ver §12. Causa identificada además: desajuste de orientación EXIF en 64 % de las cajas de moneda de val, que también invalida esa evaluación.)* | Corrección EXIF implementada (E1, §13); pendiente de reentrenar para medir su efecto |
| Falsos positivos masivos en bolígrafo | Precisión de 0.003 al evaluar con score $\ge 0.2$ | Cajas rectangulares en bolígrafos diagonales abarcan 95% de mesa vacía | Re-fotografiar bolígrafos en orientación horizontal/vertical ajustada |
| `ModuleNotFoundError: No module named 'PIL'` al lanzar `train.py` (24-nov-2026) | Fallo al importar `data.py` en `.venv_wsl` | E1 añadió `from PIL import Image` al inicio de `data.py` y `.venv_wsl` no tenía Pillow (aunque `requirements.txt` lo lista). Afecta también al modo legacy | El usuario ejecutó `pip install pillow` en WSL2. **Corregido en código (24-nov-2026, §14.0):** Pillow se importa dentro de `_leer_orientacion`; solo se exige con `APLICAR_EXIF=1` |
| Aviso de CUDA OOM transitorio | `Failed to allocate device memory of 5.43GiB` | TensorFlow intenta reservar el bloque total de memoria al iniciar | Transitorio: el asignador reintentó con bloques menores y entrenó con 5.5 GB |

## 10. Pendientes y siguientes pasos
- [x] Documentar línea base (Run 1), rebalanceo inicial (Run 2) y dataset ampliado (Run 3).
- [x] Implementar y documentar Run 4 (6 anchors por celda, escala min 0.07 y Random Zoom).
- [x] Identificar y registrar el diagnóstico detallado de las limitaciones del dataset actual.
- [x] Modificar `.gitignore` y versionar `data/` completo para análisis.
- [ ] Ajustar la captura de imágenes para monedas (tomas macro) y bolígrafos (alineación ortogonal).
- [ ] Revisar y etiquetar exhaustivamente objetos múltiples en imágenes de escritorio (evitar falsos fondos).
- [ ] Aumentar la paciencia de `ReduceLROnPlateau` (ej. de 5 a 10 épocas) para evitar estrangulamiento prematuro del LR.
- [ ] Evaluar viabilidad de Transfer Learning (`weights="imagenet"`) si los requerimientos de la materia lo permiten.
- [x] Auditorías de datos y de pipeline (24-nov-2026, §12).
- [x] E1: corrección de la orientación EXIF implementada y validada técnicamente (§13). *Sin resultados de entrenamiento todavía.*
- [x] **Run 5: entrenado y evaluado con `APLICAR_EXIF=1` por el usuario; resultado registrado en §13.2 (mAP 0.055, sin mejora).**
- [ ] Decidir el siguiente paso a la vista del Run 5 (§13.3): diagnósticos de solo lectura (evaluar con `--score` bajo; pérdida/AP sobre una muestra de train sin aumento) o un experimento E2–E6 (requiere autorización explícita).
- [x] Import de Pillow diferido en `data.py` (§14.0), con tests.
- [x] Diagnósticos de solo inferencia ejecutados en WSL2 (`d3`, `d1c`, `d6`); resultados en §14.7 y §14.9.
- [ ] **Decidir con el usuario cómo corregir BatchNorm (§14.9, §14.6): herramienta de recalibración de checkpoints y/o E6b (momentum de BN).**
- [ ] Sujeto a autorización y resultados de E1: E2 (zoom), E3 (antialias/proporción), E4 (determinismo en la asignación de anchors), E5 (cobertura de anchors, bolígrafo), E6 (dinámica del entrenamiento tras la época 11), fijar versiones del entorno.

## 11. Anexos (adjuntar o pegar)
- Salida completa de `python verificar_dataset.py` (antes y después).
- `salida/historial.csv` de cada ejecución.
- Salida de `python evaluate.py` de cada modelo.
- Captura de `nvidia-smi` con el entrenamiento en marcha.
- Resultado de `pip freeze` (versiones exactas).

## 12. Auditorías de datos y pipeline (24-nov-2026)
Las cifras de esta sección se midieron sobre el repositorio y el código actuales (no proceden de logs de entrenamiento). Scripts y visualizaciones: directorio temporal de la sesión de trabajo (no están en el repo).

### 12.1 Estructura y censo
- Imágenes `data/{train,val}/images/<clase>_<NNN>.{jpg,jpeg}` con su etiqueta `labels/<mismo nombre>.txt`, correspondencia 1:1 (1,410 de 1,410). El índice NNN es único por clase y contiguo desde 1, repartido entre train y val. Cada imagen contiene una sola clase (de 1 a 8 objetos).
- Las 1,410 imágenes se leen sin error (1,021 MPO y 389 JPEG). Tamaños variables: 1536×2048 / 2048×1536 (bolígrafo, libro, mouse, parte de laptop y moneda), 4032×3024 y 4032×2268 (laptop y moneda).
- **Orientación EXIF distinta de 1: 247 imágenes** (EXIF 3: 103, EXIF 6: 77, EXIF 8: 67), todas de moneda y laptop:

| Clase | Split | Total | Afectadas (EXIF 3 / 6 / 8) |
|---|---|---|---|
| moneda | train | 233 | 137 (26 / 59 / 52) |
| moneda | val | 67 | 40 (8 / 17 / 15) |
| laptop | train | 205 | 54 (53 / 1 / 0) |
| laptop | val | 58 | 16 (16 / 0 / 0) |
| bolígrafo, libro, mouse | ambos | 1,023 | 0 |

### 12.2 Las etiquetas están en el marco de la imagen orientada
- Prueba de aspecto (moneda con EXIF 6/8, 143 imágenes): en el marco orientado la mediana de |log AR| es ≈ 0.05–0.08 (monedas casi cuadradas); en el marco crudo, ≈ 0.57–0.59. El marco orientado es más circular en las 143 de 143.
- Revisión visual: se revisaron las 247 imágenes afectadas con las cajas dibujadas sobre el tensor que recibe el modelo tras aplicar E1; en todas las cajas encajan con el objeto. Controles sin rotación (EXIF 1 o sin etiqueta): correctos.
- `tf.io.decode_image` devuelve los píxeles **sin rotar** (comprobado en TF 2.21.0 en Windows y en TF 2.22.0rc1 en WSL2; en EXIF 3 mediante comparación de píxeles en una muestra de 12 imágenes). Hasta E1, entrenamiento y evaluación usaban píxeles sin rotar con etiquetas en el marco orientado.
- Impacto: 560 de 2,798 cajas (20 %) están en imágenes con EXIF 3/6/8 (moneda: 382 de 602 en train y 108 de 168 en val; laptop: 54 de 205 y 16 de 58). **Los cuatro runs históricos se entrenaron y evaluaron con este desajuste.**

### 12.3 Auditoría automática de las 2,798 cajas
- Sin errores de formato ni de rango (0 labels vacíos, 0 clases distintas al prefijo del archivo, 0 w/h ≤ 0, 0 valores fuera de [0,1], 0 cajas fuera de la imagen, 0 casi duplicadas). Train y val tienen distribuciones muy parecidas en todas las clases.
- Mediana del área de caja: laptop 0.52–0.54, libro 0.40, mouse 0.27–0.29, bolígrafo 0.12, moneda 0.056. Cajas por imagen: moneda 1–5 (78 imágenes con exactamente 5), bolígrafo 1–8 (77 con exactamente 8), mouse 1–3, laptop y libro 1.
- Solapes relevantes (IoU > 0.3 o cobertura > 60 % de la caja menor) en el 57–58 % de las imágenes de bolígrafo; IoU máximo 0.80. Moneda: IoU máx. 0.10; mouse 0.28; laptop y libro no aplica.
- Objetos sin anotar: no detectables automáticamente; en unas 70 vistas inspeccionadas no se vio ninguno. Dudas puntuales: `moneda_294` (objeto negro redondo sobre un teclado) y `boligrafo_218` (cajas holgadas).

### 12.4 Trazado del pipeline de datos
- No es un detector YOLO: es un SSD propio (MobileNetV2 `alpha=1.0`, `weights=None`, TF/Keras 3); "YOLO" es solo el formato de las etiquetas. Sin pesos preentrenados en el código.
- Recorrido: `decode_image` → (EXIF, solo con E1) → `tf.image.resize` 256×256 (bilineal, sin antialias, sin conservar proporción) → `/255` → aumentos solo en train (flip, zoom, brillo, contraste) → asignación de anchors → batch. Train y val usan el mismo código de carga; `evaluate.py` también.
- Coherencia píxel–caja: codificación/decodificación exacta (IoU mínimo 0.9999996); flip y zoom con IoU ≥ 0.886 en una prueba sintética de 400 casos y alineación correcta en la función real sobre 4 imágenes.

### 12.5 Entornos
- `.venv` (Windows; Run 1): Python 3.10.11, TF 2.21.0, Keras 3.12.4, NumPy 2.2.6, Pillow 12.3.0.
- `.venv_wsl` (WSL2 con GPU; runs posteriores según esta bitácora): Python 3.14.4, **TF 2.22.0rc1**, **keras-nightly 3.16.0.dev2026111903**, NumPy 2.5.5; GPU detectada. `requirements.txt` no fija versiones.

### 12.6 Hallazgos de pipeline pendientes (sin corregir; no autorizados todavía)
| Hallazgo | Estado |
|---|---|
| Cobertura de anchors débil en bolígrafo: 37 % de sus GT (406 de 1,097) sin ningún anchor IoU ≥ 0.5 y 61 % con ≤ 2; moneda 15 % y 52 %; libro 14 % / 29 %; mouse 13 % / 32 %; laptop 2 % / 10 % | Medido |
| 138 GT comparten su mejor anchor con otro GT (58 imágenes, todas de bolígrafo); la asignación usa `tensor_scatter_nd_update` con índices repetidos | Colisiones medidas; **no determinismo sin confirmar** | *(Actualización 24-nov-2026, §14.1: en CPU/Windows el resultado es determinista, gana el último GT; no confirmado aún en WSL2.)*
| `_random_zoom` descarta ~14 % de las cajas y ~4.9 % tenían ≥ 30 % del objeto visible | **Corregido con el código real (§14.2):** descarta 9.0 % de los GT cuando se aplica el zoom (4.5 % por época con p=0.5), 3.1 % con ≥ 30 % visible y **0 % con ≥ 50 % visible**. La estimación anterior sobrestimaba |
| `tf.image.resize` sin antialias (dif. media 0.013, p99 0.10 en una moneda de 12 MP) y sin conservar proporción | Medido en un caso |
| Entorno de entrenamiento con versión *release candidate* y Keras nightly sin fijar | Verificado |

## 13. Corrección controlada y estado de E1
Plan aprobado: experimentos controlados de un factor, conservando la arquitectura SSD-MobileNetV2, las 5 clases, el entrenamiento desde cero, las imágenes y etiquetas originales y los resultados históricos. Alcance autorizado hasta ahora: preparación mínima de E0 (respaldo) y **E1**. No autorizados: E2 (zoom), E3 (antialias/letterbox), E4 (asignación de anchors), E5 (diseño de anchors), E6 (dinámica del entrenamiento), cambios de arquitectura o de hiperparámetros.

### 13.1 E1: orientación EXIF — implementado y validado técnicamente
- **Qué hace (propuesta A):** lee la orientación EXIF (solo cabecera, con Pillow) y la aplica con operaciones de TensorFlow en `data.py`, **antes del resize**, en entrenamiento, validación y evaluación. Soporta las 8 orientaciones (equivalente a `PIL.ImageOps.exif_transpose`). No se modifican imágenes ni etiquetas ni se crean copias del dataset.
- **Bandera `APLICAR_EXIF`** en `config.py`: desactivada por defecto (modo legacy); se activa con la variable de entorno `APLICAR_EXIF=1`.
- **Rama y commit:** `fix/e1-exif`, commit `dd21daa` (`config.py`, `data.py`, `tests/test_exif.py`); no fusionada a `main`.
- **Verificación:**
  - `python -B -m unittest tests.test_exif`: **10/10** tests OK. (Un test detectó al principio que un valor de orientación fuera de rango se trataba como 8 en vez de como sin rotar; se corrigió el código.)
  - Rotación idéntica a PIL píxel a píxel en las 247 imágenes con EXIF 3/6/8; formas orientadas iguales a las de PIL en las 1,410. Extremo a extremo (decodificador de TF), error medio 0.0035 frente al PIL orientado y ≥ 0.0986 frente al crudo.
  - Revisión visual completa de las 247 imágenes: cajas alineadas con los objetos.
  - Modo legacy: con la bandera desactivada, 40/40 imágenes de muestra y los primeros 6 lotes de validación son idénticos a los de antes del cambio; con la bandera activa las imágenes sin rotación EXIF no cambian.
- **Precaución:** usar `APLICAR_EXIF=1` **tanto al entrenar como al evaluar** el modelo correspondiente. Evaluar con la bandera desactivada un modelo entrenado con EXIF (o al revés) usa un marco de entrada distinto y los resultados no son válidos. La bandera no se guarda con el modelo; por ahora no se añade infraestructura de configuración por run.

### 13.2 Run 5 (E1, EXIF activado): resultados
Lo ejecutó el usuario en WSL2 el 24-nov-2026 (Python 3.14.4, TF 2.22.0rc1, Keras nightly, RTX 4060); Claude no lo ejecutó. Comandos: `APLICAR_EXIF=1 python train.py --epocas 150` y `APLICAR_EXIF=1 python evaluate.py --score 0.5`. **Único factor cambiado respecto a Run 4:** la orientación EXIF. Primer intento fallido por falta de Pillow en `.venv_wsl` (§9); el entrenamiento se lanzó de nuevo tras `pip install pillow`.

**Fuentes (archivos de este repo):** `archivo/historial_m5.csv` (copia de `salida/historial.csv`, SHA256 `f64e50e5…1337b`), `archivo/evaluate_m5.txt` (transcrito de la salida pegada por el usuario) y `archivo/best_m5.keras` (SHA256 `0f092594775bc95980befcbee367046c91e7a0fdfcd9236506153607ead93526`, idéntico a `salida/best_model.keras`). Una limitación: el log de la consola no imprime el valor de `APLICAR_EXIF`; que estuvo activa se desprende del comando indicado por el usuario, no de la salida.

**Resumen:** 38 épocas (EarlyStopping, paciencia 30 desde la época 8) · mejor época **8**, `val_loss` **7.9220** · `loss` final 2.5638, `val_loss` final 14.4109 · LR 1e-3 hasta la época 13; `ReduceLROnPlateau` redujo el LR tras las épocas 13, 18, 23, 28, 33 y 38 (a 5e-4, 2.5e-4, 1.25e-4, 6.25e-5, 3.125e-5 y 1.56e-5; este último no llegó a usarse) · ~12–14 s por época desde la 2 (la 1 tardó 97 s por la compilación XLA).

**Resultados de `evaluate.py` (score ≥ 0.5, `val`):** mAP@0.5 **0.055**; Precision 0.000; Recall 0.000.

| Clase | AP@0.5 Run 5 | GT | AP Run 4 (tabla §6) |
|---|---|---|---|
| boligrafo | 0.001 | 250 | 0.001 |
| laptop | 0.163 | 58 | 0.159 |
| libro | 0.109 | 88 | 0.110 |
| moneda | 0.000 | 168 | 0.000 |
| mouse | 0.003 | 60 | 0.003 |

**Salida de cada época** (`archivo/historial_m5.csv`; época = columna `epoch` + 1; `loss` con aumentos, `val_loss` sin aumentos):

| Época | LR | `loss` | `val_loss` | |
|---|---|---|---|---|
| 1 | 1e-3 | 8.5523 | 10.1540 | |
| 2 | 1e-3 | 5.9869 | 9.6014 | |
| 3 | 1e-3 | 5.5853 | 9.0855 | |
| 4 | 1e-3 | 5.4587 | 8.7872 | |
| 5 | 1e-3 | 5.3795 | 8.5510 | |
| 6 | 1e-3 | 5.2473 | 8.1920 | |
| 7 | 1e-3 | 5.0301 | 7.9796 | |
| 8 | 1e-3 | 4.8737 | 7.9220 | **mejor (checkpoint guardado)** |
| 9 | 1e-3 | 4.6711 | 8.0754 | |
| 10 | 1e-3 | 4.4002 | 8.8592 | |
| 11 | 1e-3 | 4.2338 | 9.2328 | |
| 12 | 1e-3 | 4.1089 | 9.3855 | |
| 13 | 1e-3 | 3.9707 | 10.9500 | |
| 14 | 5e-4 | 3.6766 | 10.3016 | |
| 15 | 5e-4 | 3.5243 | 10.2533 | |
| 16 | 5e-4 | 3.4159 | 10.4466 | |
| 17 | 5e-4 | 3.3069 | 11.4778 | |
| 18 | 5e-4 | 3.3364 | 11.4087 | |
| 19 | 2.5e-4 | 3.1755 | 11.8260 | |
| 20 | 2.5e-4 | 3.1139 | 12.1146 | |
| 21 | 2.5e-4 | 3.0651 | 12.5152 | |
| 22 | 2.5e-4 | 3.0192 | 12.4137 | |
| 23 | 2.5e-4 | 3.0041 | 13.3713 | |
| 24 | 1.25e-4 | 2.9110 | 13.2581 | |
| 25 | 1.25e-4 | 2.8870 | 13.2646 | |
| 26 | 1.25e-4 | 2.7879 | 13.4971 | |
| 27 | 1.25e-4 | 2.7845 | 13.5389 | |
| 28 | 1.25e-4 | 2.7714 | 13.7511 | |
| 29 | 6.25e-5 | 2.6999 | 13.8250 | |
| 30 | 6.25e-5 | 2.7353 | 13.8601 | |
| 31 | 6.25e-5 | 2.6865 | 13.8205 | |
| 32 | 6.25e-5 | 2.6199 | 14.0333 | |
| 33 | 6.25e-5 | 2.5825 | 14.2723 | |
| 34 | 3.125e-5 | 2.6584 | 14.2434 | |
| 35 | 3.125e-5 | 2.5752 | 14.3198 | |
| 36 | 3.125e-5 | 2.6042 | 14.3625 | |
| 37 | 3.125e-5 | 2.5816 | 14.3496 | |
| 38 | 3.125e-5 | 2.5638 | 14.4109 | |

### 13.3 Lectura del Run 5 (hechos, hipótesis y lo que no se sabe)
**Hechos (de los datos de arriba):**
- E1 **no mejoró el mAP** (0.055 → 0.055) ni el AP de moneda (0.000). Por clase, las diferencias con Run 4 son ≤ 0.004.
- La `loss` de entrenamiento baja de forma continua (8.55 → 2.56) mientras la `val_loss` toca su mínimo en la época 8 (7.92) y sube hasta 14.41: brecha final de ≈ 5.6×. La subida empieza con el LR todavía en 1e-3 (épocas 9–13: 8.08, 8.86, 9.23, 9.39, 10.95); las reducciones del LR (desde la época 13) llegan después y no la revierten. Es el mismo patrón de los runs anteriores (óptimo hacia la época 11 en Run 4).
- Solo laptop y libro (objetos grandes, 1 por imagen) alcanzan AP > 0.1; bolígrafo, moneda y mouse están en ≈ 0.
- Con score ≥ 0.5 no hay ninguna detección correcta (precisión y recall 0.000). No se evaluó con score bajo en este run.

**[REVISADO en §14.9: esta conclusión quedó contaminada por el desfase de las estadísticas de BatchNorm; con BN recalibrado moneda alcanza AP 0.334 con EXIF frente a 0.032 sin EXIF.]** **Consecuencia (original) para las hipótesis de §7 y §12:** la hipótesis de que el desajuste EXIF explicaba el AP ≈ 0 de moneda **no se sostiene** (moneda tenía 63–64 % de sus cajas desalineadas y sigue en 0.000 con la corrección). El desajuste era un defecto real y está corregido, pero no es la causa principal de la baja fiabilidad.

**Cautelas:**
- La `val_loss` de Run 5 no es comparable con la de los Runs 1–4 (cambió el marco de entrada de val); 7.92 frente a 7.54 no indica empeoramiento ni mejora atribuibles.
- Una sola ejecución: sin repeticiones no se puede estimar la variabilidad entre runs (semilla, y posible no determinismo de la asignación, §12.6); diferencias de ~0.004 de AP no son interpretables.
- No se conserva la curva de métricas por época (solo `loss` y `val_loss`), de modo que no se sabe cómo evoluciona el mAP durante el entrenamiento.

**Hipótesis abiertas (no probadas; ningún cambio autorizado; ver la corrección de §14.7 sobre el sobreajuste):** sobreajuste por capacidad frente a datos (2.7 M de parámetros desde cero con 1,097 imágenes); inestabilidad con LR 1e-3 y BatchNorm (E6); cobertura y asignación de anchors, sobre todo en bolígrafo (E4, E5); descarte de cajas por el zoom (E2); resize sin antialias (E3).

### 13.4 Respaldos y archivos del Run 5
- `archivo/best_m5.keras` ahora contiene el modelo del **Run 5** (SHA256 `0f092594…93526`; mismo que `salida/best_model.keras`). Se sobrescribió por indicación del usuario; la copia anterior (checkpoint de Run 4 según esta bitácora, SHA256 `bb30e9fdc872f27b52b76582b50efd58beabb64f10dd5eeb094626c839fcbba8`) sigue recuperable en el commit `794a049` (`git show 794a049:archivo/best_m5.keras`).
- `best_m1..m4.keras` no se modificaron (`best_m4.keras` = Run 3 según §5).
- `archivo/historial_m5.csv` y `archivo/evaluate_m5.txt`: ver §13.2. Convención hacia delante: el sufijo `_mN` identifica el run (`best_mN.keras`, `historial_mN.csv`, `evaluate_mN.txt`); archivar antes de volver a entrenar, porque `train.py` sobrescribe `salida/best_model.keras` y `salida/historial.csv`.

### 13.5 Regla de trabajo
Claude no ejecuta entrenamientos; los lanza el usuario manualmente desde WSL2 (ver `CLAUDE.md`, 9.7).

## 14. Cambio de Pillow y diagnóstico posterior al Run 5 (24-nov-2026)
Todo lo de esta sección es de **solo lectura** sobre el código y los datos: no se entrenó, no se cambió ningún componente del pipeline (anchors, zoom, resize, pérdida) y no se tocaron imágenes ni etiquetas. Scripts y salidas: `diagnostico/run5/`.

### 14.0 Pillow como dependencia condicional (cambio de código autorizado)
- **Cambio** (`data.py`): se quitó `from PIL import Image` del inicio del módulo; ahora se importa dentro de `_leer_orientacion`, **antes** del `try` que lee la cabecera. Con `APLICAR_EXIF=0` nunca se llama a esa función, así que el pipeline legacy no necesita Pillow. Con `APLICAR_EXIF=1`, si Pillow no está instalado se lanza `ImportError: APLICAR_EXIF=1 necesita Pillow (pip install pillow).` (el import quedó fuera del `try` para que no se trate toda imagen como "sin rotar" en silencio). La orientación con TensorFlow (`_aplicar_orientacion`) no cambió.
- **Tests** (`tests/test_exif.py`, ahora **15/15 OK**, 10 anteriores + 5 nuevos que arrancan un intérprete nuevo y simulan Pillow ausente con `sys.modules['PIL'] = None`): (1) legacy sin Pillow, con la bandera sin definir y en `0`: importa `data`, lista muestras, construye `dataset_solo_imagenes`, `construir_dataset('val')` y `construir_dataset('train', entrenamiento=True)` (con aumentos) y obtiene un lote; (2) la imagen legacy es byte a byte idéntica con y sin Pillow (SHA256); (3) `APLICAR_EXIF=1` sin Pillow falla con el mensaje claro; (4) `APLICAR_EXIF=1` con Pillow lee las orientaciones reales de `val` (`[1, 3, 6, 8]`); (5) análisis estático (`ast`): `data.py` no importa Pillow a nivel de módulo. Ejecutados en Windows (`.venv`, Python 3.10.11, TF 2.21.0).
- No se verificó todavía en `.venv_wsl` (sin acceso desde aquí; el usuario ya instaló Pillow allí).

### 14.1 Asignación de anchors y colisiones (`diagnostico/run5/d1_anchors.py`, `d1_anchors_salida.txt`, `d1b_hash_asignacion.py`)
- **Hecho:** una réplica en numpy reproduce la salida real de `_codificar_objetivos` (clases **y** offsets) en 1,097/1,097 imágenes de train y 313/313 de val cuando, ante índices repetidos en `tensor_scatter_nd_update`, **gana el último GT**; la variante "gana el primero" discrepa en exactamente las 45 (train) + 13 (val) imágenes con colisión. En CPU (Windows, TF 2.21.0) la asignación es **determinista** (misma huella SHA256 en dos ejecuciones: `5c99554a…09f1`). Como la asignación corre dentro de `tf.data` (CPU), el no determinismo temido en §12.6 no se observa; **falta confirmarlo con TF 2.22.0rc1 en WSL2** (comando en §14.5).
- **Colisiones (dos GT con el mismo mejor anchor):** 58 imágenes, todas de bolígrafo. GT que comparten anchor: 12.8 % de los de bolígrafo en train (12.0 % en val). GT que terminan **sin ningún anchor positivo**: 3.7 % de bolígrafo en train (2.0 % en val); el resto de clases 0 %. El GT que pierde el desempate no recibe su anchor forzado.
- **Positivos por imagen:** media 5.1 (mín. 1, máx. 29) de 2,046 anchors (≈ 0.25 %); ninguna imagen sin positivos.
- **GT sin ningún anchor con IoU ≥ 0.5 (train):** bolígrafo 37.3 %, moneda 14.5 %, mouse 14.6 %, libro 13.6 %, laptop 2.0 %. A esos GT solo los cubre su mejor anchor forzado, con IoU mediana 0.41 (bolígrafo) a 0.47 (moneda, mouse). Con ≤ 2 anchors: bolígrafo 61.5 %, moneda 51.3 %, mouse 35.8 %, libro 28.2 %, laptop 10.7 %. Anchors positivos por GT (tras colisiones): bolígrafo 2.1, moneda 2.5, mouse 3.0, libro 3.1, laptop 3.4.
- **Positivos por nivel del mapa (train, sin aumentos; total 5,552):** el nivel 16×16 (1,536 de los 2,046 anchors, 75 %) aporta solo 172 positivos (3.1 %; casi todos de moneda: 162). Bolígrafo: 8×8 → 820, 4×4 → 714, 2×2 → 123, 1×1 → 85 (de 1,750). Moneda: 8×8 → 1,216 de 1,530. Laptop: 1×1 → 600 de 696; libro: 1×1 → 641 de 941 (los objetos grandes dependen de la celda única). Los anchors 3:1 y 1:3 aportan 33 % de los positivos de bolígrafo (576 de 1,750). Con zoom real (p=0.5, 3 pasadas) la distribución por clase es parecida, con más positivos en niveles profundos.
- **Lectura:** la cobertura de bolígrafo es el defecto de asignación más claro (y cuantificado). **Hipótesis (no probada):** el 75 % de los anchors (nivel 16×16, escala 0.07–0.25) casi nunca es positivo, de modo que el diseño de anchors no se ajusta a los tamaños reales (mediana de caja ≈ 60 px en moneda; ver `d5_umbral_iou.txt` para el efecto de un umbral de IoU menor).

### 14.2 Random Zoom (`d2_zoom.py`, `d2_zoom_salida.txt`)
- **Método:** réplica instrumentada de `_random_zoom` (devuelve además el recorte); **validada contra la función real**: 400/400 casos idénticos (cajas e imagen) con la misma semilla. Se midieron 10,870 GT-observaciones de train con el zoom aplicado (p=1.0; en entrenamiento p=0.5, así que los porcentajes por época son la mitad).
- **Descarte:** 9.0 % de los GT con zoom aplicado (≈ 4.5 % por época): bolígrafo 9.5 %, moneda 16.7 %, mouse 6.4 %, laptop 0 %, libro 0 %. **Siempre es por centro fuera del recorte**; nunca por tamaño mínimo (0.02). Descartados con ≥ 30 % del objeto visible: 3.1 %; **con ≥ 50 % visible: 0.0 %**. Esto **corrige** la estimación anterior de §12.6 (14 % y 4.9 %, hecha con una réplica no validada).
- **Imágenes:** el zoom se centra en un objeto, así que ninguna imagen se queda sin objetos (0.0 %). Objetos conservados / originales: bolígrafo 0.91, moneda 0.83, mouse 0.94, laptop y libro 1.00.
- **Cajas recortadas pero conservadas** (la caja se recorta al borde del recorte, correcta para la parte visible): con < 80 % del objeto visible: laptop 55 %, libro 37 %, mouse 25 %, bolígrafo 23 %, moneda 13 %; con < 50 %: 0.7–3.6 %.
- **Escala:** mediana de moneda 60 → 77 px (×1.3); bolígrafo 96×114 → 116×143 px; laptop 187 → 229 px.
- **Dato de código:** el zoom se aplica **después** del resize a 256×256, así que amplía una imagen ya reducida (interpolación; no añade detalle).
- **Lectura:** el zoom no destruye objetos visibles en proporción relevante; E2 (corregir el descarte) tiene **poco que corregir**. Su efecto neto sobre el rendimiento sigue sin aislarse (en Run 4 se cambió junto a los anchors).

### 14.3 Curvas de aprendizaje del Run 5 (`d4_curvas.py`, `d4_curvas.txt`, `curvas_run5.png`)
- Mejor época 8 (val_loss 7.9220, loss 4.8737). Primera época posterior que supera el mínimo en +2 % / +10 % / +25 % / +50 % / +75 %: 10 / 10 / 13 / 20 / 32.
- Cociente val_loss/loss: 1.19 (época 1), 1.63 (8), 2.01 (10), 2.76 (13), 4.45 (23), 5.62 (38). La `loss` de train incluye aumentos (es más difícil), por lo que la brecha real es mayor que la visible.
- Ritmo por época: épocas 1→8 `loss` −0.53, `val_loss` −0.32; 8→13 (LR 1e-3) `loss` −0.18, `val_loss` **+0.61**; 13→23 `val_loss` +0.24; 23→38 `val_loss` +0.07. Tras la época 8, `val_loss` sube en 22 de 30 pasos; correlación loss–val_loss en esas épocas −0.975.
- **Hecho:** el deterioro empieza con LR = 1e-3 constante (épocas 9–13). `ReduceLROnPlateau(patience=5)` solo puede actuar 5 épocas después de la mejor (época 13), cuando `val_loss` ya estaba +38 % sobre el mínimo; tras reducir el LR baja un poco (10.95 → 10.30) y nunca vuelve cerca del mínimo. Con LR ≤ 1.25e-4 el modelo sigue memorizando (loss baja) y val_loss sigue subiendo: no es solo un LR demasiado bajo.
- **[Corregido en §14.7: en el checkpoint de la época 8 no hay brecha train–val (7.99 frente a 7.92 sin aumentos); el "sobreajuste" no está demostrado y la brecha del historial puede deberse a BatchNorm.]** Hipótesis original (sin probar): sobreajuste por memorización (2.7 M de parámetros desde cero, 1,097 imágenes) con inicio de la divergencia ligado a la etapa de LR alto. No se puede separar de la inestabilidad (BatchNorm/LR) con estos datos; solo se guardó el checkpoint de la época 8, así que no se pueden evaluar las épocas posteriores.

### 14.4 `val_loss` frente a mAP
- **Hechos:** (a) en los Runs 1–4 (bitácora) la `val_loss` mejoró (9.26 → 8.10 → 7.54) mientras el mAP cayó (0.065 → 0.061 → 0.055) y en el Run 5 se mantiene en 0.055; (b) el checkpoint se elige por `val_loss`, y no se guarda el mAP por época, así que no se sabe si el mejor mAP ocurre en otra época; (c) `perdida_ssd` suma clasificación en positivos, solo los 3 negativos más difíciles por positivo, y regresión, todo dividido entre el nº de positivos: no evalúa el ranking de scores ni el IoU ≥ 0.5 sobre los que se construye el AP; (d) con score ≥ 0.5 no hay ninguna detección (precisión y recall 0.000) pese a una `val_loss` de 7.9: el modelo no alcanza confianza alta en ninguna caja.
- **Qué falta para explicarlo:** descomponer la pérdida (cls positivos / negativos minados / caja) por clase, medir en train y val si el fallo es de localización o de clasificación (AP agnóstico de clase, matriz de confusión, recall potencial), la brecha train–val del mismo checkpoint y las probabilidades en los positivos. Todo eso lo calcula `diagnostico/run5/d3_modelo.py` (solo inferencia; **no se pudo ejecutar sobre los checkpoints en Windows**: Keras 3.12.4 no puede deserializar un `.keras` guardado con Keras nightly 3.16). Se verificó el script con un modelo sin entrenar: su réplica de la pérdida coincide con `perdida_ssd` (diferencia ≈ 3e-7). **Ejecutado por el usuario en WSL2: resultados en §14.7.**

### 14.5 Comandos para ejecutar en WSL2 (solo inferencia; no entrenan)
```bash
cd /mnt/d/Respaldo/Documents/Escuela/InteligenciaArtificial/Proyecto/Modelo-IA-AR && source .venv_wsl/bin/activate
python diagnostico/run5/d1b_hash_asignacion.py        # debe dar 5c99554adccc...09f1 (el mismo hash que en Windows)
APLICAR_EXIF=1 python diagnostico/run5/d3_modelo.py --modelo archivo/best_m5.keras --etiqueta run5
git show 794a049:archivo/best_m5.keras > /tmp/run4.keras
python diagnostico/run5/d3_modelo.py --modelo /tmp/run4.keras --etiqueta run4    # Run 4: sin EXIF
```
**Segunda ronda (solo pasadas hacia adelante, sin gradientes; las ejecuta el usuario):**
```bash
python diagnostico/run5/d1c_asignacion_npz.py --etiqueta wsl
python diagnostico/run5/d1c_asignacion_npz.py --comparar diagnostico/run5/d1c_windows.npz diagnostico/run5/d1c_wsl.npz
APLICAR_EXIF=1 python diagnostico/run5/d6_batchnorm.py --modelo archivo/best_m5.keras --etiqueta run5
python diagnostico/run5/d6_batchnorm.py --modelo /tmp/run4.keras --etiqueta run4
python evaluate.py --modelo /tmp/run4.keras --score 0.5     # cifra oficial de mAP del Run 4 (sin APLICAR_EXIF)
```
`d6` tarda unos minutos y escribe `d6_<etiqueta>.txt`; recalibra las estadísticas de BatchNorm solo en memoria (no guarda en `archivo/` ni en `salida/`).

Cada `d3` escribe `diagnostico/run5/d3_<etiqueta>.txt` y tarda unos minutos. El Run 4 permite comprobar la `val_loss` 7.5415 de la bitácora y comparar el AP en un mismo conjunto de métricas.

### 14.6 Propuestas de experimentos (sin autorización; a decidir; **reordenadas tras §14.9**)
1. **Corrección de BatchNorm (prioridad 1; evidencia en §14.9).** Dos piezas independientes:
   - **1a. Herramienta de recalibración de checkpoints** (**implementada y probada: E6a, §14.10**; pendiente de ejecución del usuario sobre checkpoints reales; nuevo script, no cambia entrenamiento ni código existente): recalcula las estadísticas móviles de BN con los pesos intactos y guarda un `.keras` nuevo. Permite evaluar/exportar los checkpoints ya entrenados con mAP ≈ 0.28–0.31 en vez de 0.055–0.065.
   - **1b. E6b: momentum de BN del backbone 0.999 → 0.99** (un factor; cambio pequeño en `model.py`) y/o seleccionar el checkpoint con `val_loss` calculada con estadísticas ya convergidas. Requiere entrenar (lo hace el usuario).
2. **Re-entrenar con la corrección de BN y comparar con y sin EXIF** (un factor por vez): los dos únicos checkpoints disponibles (épocas 8 y 11, de runs distintos) no permiten aislar el efecto del EXIF.
3. **E5a: umbral de IoU positivo 0.5 → 0.4** (una línea de `config.py`; ver `d5_umbral_iou.txt`: bolígrafo sin anchor 37.3 % → 17.9 %). Sigue siendo razonable, pero debe medirse **después** de corregir BN.
Por ahora no se justifican E6c (antes "E6a": `--lr 3e-4`; la etiqueta E6a pasó a la recalibración de BN, §14.10; su motivación era un sobreajuste no demostrado) ni E2a (zoom). Fuera del plan, decisión del usuario: transfer learning.

### 14.7 Resultados de `d3` en WSL2 (checkpoints Run 5 y Run 4; solo inferencia) — `d3_run5.txt`, `d3_run4.txt`
> **Nota (§14.9):** todas las cifras de clasificación, confusión de clase, recall y AP de este apartado se midieron con BatchNorm en modo inferencia con estadísticas móviles desfasadas, que §14.9 demuestra que distorsionan el resultado; **no deben usarse para sacar conclusiones sobre la capacidad del modelo**. Siguen siendo válidas la reproducción exacta de la `val_loss` y la ausencia de brecha train–val en ese modo. La alerta del hash quedó resuelta en §14.9.
Ejecutado por el usuario el 24-nov-2026 (TF 2.22.0-rc1, GPU). Los `.txt` están en `diagnostico/run5/`.
- **Reproducibilidad verificada:** la `val_loss` recalculada coincide con el historial y con la bitácora: Run 5 con `APLICAR_EXIF=1`: **7.9221** (historial 7.9220); Run 4 sin EXIF: **7.5415** (bitácora 7.5415). La réplica por componentes coincide con `perdida_ssd` (diferencia < 3e-8). Esto confirma que `archivo/best_m5.keras` (Run 5) es el checkpoint de la época 8 y que el checkpoint antiguo recuperado de `794a049` es el del Run 4, y refuerza que el Run 5 se entrenó con la bandera activa (con la bandera desactivada el marco de val sería otro).
- **Hallazgo principal: en el mejor checkpoint no hay brecha train–val.** Pérdida sin aumentos en modo inferencia: Run 5 train **7.989** frente a val **7.922**; Run 4 train 7.530 frente a val 7.542. El modelo rinde igual en entrenamiento que en validación: **no está sobreajustado en la época 8** (se parece más a un modelo que no ajusta). **Corrige** la lectura de §13.3 y §14.3 como "sobreajuste": la brecha del historial (loss 4.87 frente a val_loss 7.92 en la época 8) **no es una brecha de generalización**, porque el mismo checkpoint da 7.99 sobre train sin aumentos. La `loss` del historial es la de entrenamiento (con aumentos y BatchNorm con estadísticas del lote).
- **Hipótesis nueva (por explicar, sin probar): BatchNorm.** Los BN del backbone MobileNetV2 usan `momentum=0.999` (verificado en `keras.applications`); con 69 pasos por época, tras 8 épocas las estadísticas móviles conservan 0.999^552 = 58 % de su valor inicial (47 % en la época 11; 7 % en la 38). Eso podría explicar que el modo inferencia (medias móviles) rinda mucho peor que el modo entrenamiento. **`d6_batchnorm.py`** lo comprueba sin entrenar (pérdida con/sin aumentos y BN de inferencia/de lote; recalibración de las estadísticas móviles con los pesos intactos y `evaluate.py` antes/después). **Pendiente de ejecución por el usuario (§14.5).** Si se confirma, la `val_loss` y el mAP de todos los runs estarían subestimando el modelo.
- **Calidad de la clasificación en los anchors positivos (val):** probabilidad media de la clase correcta 0.03 (bolígrafo) a 0.27 (laptop) en Run 5; probabilidad de fondo entre 0.47 y 0.81; pérdida de clasificación por positivo entre 1.9 y 4.0 (una distribución uniforme entre 6 clases daría 1.79; bolígrafo está por encima). El modelo no discrimina bien las clases.
- **Localización frente a clasificación (val, GT con alguna detección de IoU ≥ 0.5 de cualquier clase, "recall potencial"):** Run 5: bolígrafo 90/250 (36 %), laptop 53/58, libro 57/88, moneda 47/168 (28 %), mouse 46/60. AP agnóstico de clase: 0.097 (Run 5), 0.112 (Run 4): también la localización/objectness es débil.
- **Confusión de clase (de ese recall potencial):** moneda: **43 de 47 se etiquetan como mouse**, ninguna como moneda (Run 4: 36 de 42). Libro → laptop en Run 5 (52 de 57); laptop → libro en Run 4 (53 de 56): para los dos objetos grandes el modelo elige una sola de las dos clases, y cuál cambia entre runs. Bolígrafo → mouse en 47 de 90. Es decir, la moneda **sí se localiza** en parte y falla la clasificación.
- **Descomposición de la pérdida de val por positivo (cls_pos + cls_neg + caja):** Run 5 2.91 + 2.52 + 2.71; Run 4 3.39 + 1.91 + 2.73. La `val_loss` menor del Run 4 viene del término de negativos (menos "falsos objetos": 1–2 anchors negativos con probabilidad de objeto > 0.5 por imagen frente a ≈ 21 en Run 5), no de acertar mejor las clases. Esto explica en parte por qué `val_loss` y mAP no se alinean (hecho; la interpretación es una hipótesis).
- **Precisión/recall por umbral (val):** con score ≥ 0.5 ninguna detección en ambos runs. Run 5 con score ≥ 0.3: 313 detecciones (una por imagen), 51 TP (precisión 0.163, recall 0.082); con score ≥ 0.1: recall 0.277 con precisión 0.003.
- **mAP medido por `d3` frente a `evaluate.py`:** Run 5: 0.057 (d3) frente a 0.055 (`evaluate.py`); Run 4: 0.065 (d3) frente a 0.055 (bitácora). `d3` reordena las detecciones empatadas con un `argsort` no estable, por lo que su mAP puede diferir unas milésimas de `evaluate.py`; **la diferencia del Run 4 (0.010) está sin resolver**: se pide ejecutar `evaluate.py` sobre el checkpoint de Run 4 (§14.5). No usar el mAP de `d3` como cifra oficial.
- **Asignación de anchors entre entornos — alerta:** la huella SHA256 de la asignación en WSL (TF 2.22.0-rc1) es `a2d37ede…fa913`, distinta de la de Windows (`5c99554a…09f1`). El hash redondea los offsets a 5 decimales, así que una diferencia de coma flotante podría bastar; **no está establecido** si cambia el conjunto de anchors positivos o las clases. `d1c_asignacion_npz.py` compara sin redondeo (§14.5). Mientras tanto, la coincidencia exacta de la `val_loss` de val (que depende de la asignación) en WSL con el historial sugiere que la asignación es coherente dentro de WSL.

### 14.8 Incidente: entrenamiento lanzado por error (24-nov-2026, ~12:43)
- **Qué pasó:** al registrar §14.6 usé un heredoc de Bash sin comillas; el texto `` `python train.py --lr 3e-4` `` fue interpretado como sustitución de comandos y **se ejecutó `train.py --lr 3e-4` en Windows (CPU, sin `APLICAR_EXIF`)**. Fue un error mío y **contraviene la regla de no entrenar** (CLAUDE.md 9.7); no fue una decisión deliberada.
- **Alcance:** el proceso corrió unos 2 minutos y terminó la época 1 (loss 10.1627, val_loss 10.7910, LR 3e-4); sobrescribió `salida/best_model.keras` y `salida/historial.csv` (el Run 5 original). Lo detuve con la herramienta de tareas (el bloqueo de permisos impidió terminar el proceso por otras vías).
- **Reparación (verificada):** `salida/best_model.keras` y `salida/historial.csv` se restauraron desde `archivo/best_m5.keras` y `archivo/historial_m5.csv`, y sus SHA256 coinciden con los originales (`0f092594…93526` y `f64e50e5…1337b`). Ningún proceso Python quedó en ejecución. No se tocaron `data/`, `archivo/` ni el código; la época 1 del entrenamiento accidental **no es un resultado del proyecto** y no se usa.
- **Medida:** los bloques de texto con comandos se escriben desde ahora con heredocs entrecomillados (`<<'EOF'`) o con la herramienta de escritura de archivos.

### 14.9 BatchNorm confirmado como causa de la métrica baja — `d6_run5.txt`, `d6_run4.txt`, `d1c_*.npz`
Ejecutado por el usuario en WSL2 (solo pasadas hacia adelante; los pesos entrenables no cambian y no se guarda nada en `archivo/` ni `salida/`).
- **Asignación de anchors idéntica entre entornos (alerta de §14.7 resuelta):** Windows (TF 2.21.0) y WSL (TF 2.22.0-rc1): 7,151 anchors positivos en los dos, 0 pares (imagen, anchor) distintos, 0 clases distintas, diferencia máxima de offset 9.5e-7. La diferencia de hash era solo el redondeo.
- **A. La pérdida de entrenamiento del historial sí se reproduce con BN de lote.** Run 5 (época 8): train sin aumentos 8.017 (BN de inferencia) frente a **4.791** (BN con estadísticas del lote); train con aumentos 7.767 frente a **4.806** (historial: 4.874); val 8.099 frente a 4.953. Run 4: 7.818 → 4.612 (train), 7.936 → 4.759 (val). Las estadísticas de lote dan ≈ 4.7–4.9 con o sin aumentos, y las móviles ≈ 7.8–8.1: **la diferencia es de BN, no de aumentos ni de sobreajuste.**
- **B. Recalibrar las estadísticas móviles de BN (momentum 0.9, 3 pasadas sobre train sin aumentos; pesos intactos) y evaluar con `evaluate.py` (score 0.5):**

| Checkpoint | `val_loss` (inferencia) | mAP original | mAP recalibrado | bolígrafo | laptop | libro | moneda | mouse |
|---|---|---|---|---|---|---|---|---|
| Run 5 (EXIF, época 8) | 8.099 → **4.917** | 0.062* | **0.283** | 0.053 | 0.519 | 0.294 | **0.334** | 0.216 |
| Run 4 (sin EXIF, época 11) | 7.937 → **4.684** | 0.065* | **0.311** | 0.188 | 0.683 | 0.389 | 0.032 | 0.266 |

  \* En esta ejecución, `evaluate.py` sobre el modelo original re-guardado dio 0.062 (Run 5) y 0.065 (Run 4), mientras que `evaluate.py` directo dio 0.055 (Run 5, ejecución anterior) y 0.056 (Run 4). La diferencia (≤ 0.01) **no está explicada** (ruido numérico de GPU/XLA con puntuaciones casi empatadas, o el re-guardado); es pequeña frente al efecto de la recalibración. Precisión/recall tras recalibrar (score ≥ 0.5): Run 5 1.000 / 0.008; Run 4 0.755 / 0.059; con score ≥ 0.2: Run 5 0.175 / 0.381; Run 4 0.276 / 0.345. Las estadísticas móviles cambiaron mucho: cambio relativo mediano de la media por capa 0.68 (Run 5) y 0.60 (Run 4), máximo 0.88 (`Conv_1_bn`).
- **Hechos:**
  1. Con los **mismos pesos**, recalibrar las estadísticas de BN multiplica el mAP por ≈ 5 (0.055 → 0.283; 0.065 → 0.311) y baja la `val_loss` de 8 a ≈ 4.7–4.9. Los pesos entrenados sí contienen información útil; el cuello de botella de la métrica era el modo inferencia de BN. El objetivo de mAP ≥ 0.50 sigue sin alcanzarse, pero la distancia pasa de ≈ 0.45 a ≈ 0.2.
  2. La causa está en el entrenamiento: BN del backbone con `momentum=0.999` (verificado) y solo 69 pasos por época (tras 552 pasos, 58 % del valor inicial).
  3. **Efecto EXIF en moneda:** con BN recalibrado, moneda tiene AP **0.334 con EXIF (Run 5) frente a 0.032 sin EXIF (Run 4)**. Esto contradice la conclusión de §13.3 ("el EXIF no era la causa principal"), que se midió bajo BN desfasado y queda **revisada**: los datos actuales son compatibles con que el desajuste EXIF fuera la causa principal del fallo en moneda.
  4. Bolígrafo, laptop, libro y mouse están mejor en el Run 4 (época 11, sin EXIF) que en el Run 5 (época 8, con EXIF); mAP 0.311 frente a 0.283.
- **Cautelas:** son dos checkpoints de runs distintos, en épocas distintas (8 y 11), cada uno con un solo run y con ruido de medida de ≈ 0.01; no aíslan el efecto del EXIF (sí lo hace parcialmente la diferencia de moneda, mucho mayor que el ruido). El mejor checkpoint real es desconocido: se eligió por `val_loss` con BN desfasado, y el entrenamiento siguió hasta la época 38 con la `loss` bajando (4.87 → 2.56); las épocas posteriores podrían ser mejores o peores con BN correcto, pero no se guardaron. Con BN recalibrado, la brecha train–val es pequeña (Run 5: 4.70 frente a 4.92), así que tampoco hay evidencia de sobreajuste en esa época.
- **Efecto sobre conclusiones anteriores:** (a) las comparaciones de mAP entre Runs 1–4 (0.065 → 0.061 → 0.055) y la idea de que la `val_loss` y el mAP "no están alineados" se midieron con BN desfasado y deben reevaluarse; (b) el diagnóstico de clasificación pobre y de confusión moneda→mouse de §14.7 queda en entredicho; (c) §14.1–§14.2 (anchors y zoom) no dependen de BN y siguen vigentes.

### 14.10 E6a: herramienta de recalibración de BN, comparación normalizada y discrepancia de evaluación (24-nov-2026)
Autorizado por el usuario: solo la herramienta de recalibración y sus pruebas (**etiqueta E6a**; el antiguo "E6a" de §14.6, `--lr 3e-4`, pasa a **E6c**, descartado por ahora). No se cambió el momentum (E6b) ni se entrenó.

**14.10.1 Condiciones de cada evaluación (solo se mezclan filas con las mismas condiciones; los grupos se mantienen separados)**

*Qué mide cada umbral (hecho, de `evaluate.py`):* el **umbral de IoU** es 0.5 en todos los resultados (valor fijo de `evaluar`, sin opción de línea de comandos). El **umbral de score no entra en el AP/mAP**: `predecir` conserva toda detección con score > 0.01 y el AP se calcula con ellas; `--score` solo fija el corte de **precisión y recall**. Comprobado con los datos: el mismo checkpoint recalibrado da mAP 0.283 con score 0.5 y con 0.2 (Run 5), y 0.311 con ambos (Run 4). Por tanto los mAP de la tabla son comparables en cuanto a umbrales; las diferencias entre filas vienen de otras causas.

| Id | Checkpoint | BN | EXIF (entreno / evaluación) | score | IoU | bolígrafo | laptop | libro | moneda | mouse | **mAP** | P / R | Fuente |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Grupo A: BN original (estadísticas móviles del entrenamiento)** | | | | | | | | | | | | | |
| A1 | Run 5, época 8 (`archivo/best_m5.keras`, SHA `0f092594…93526`) | original | sí / sí | 0.5 | 0.5 | 0.001 | 0.163 | 0.109 | 0.000 | 0.003 | **0.055** | 0.000 / 0.000 | `evaluate.py` directo (`archivo/evaluate_m5.txt`) |
| A2 | Run 5, época 8, **re-guardado** antes de evaluar | original | sí / sí | 0.5 | 0.5 | 0.001 | 0.189 | 0.117 | 0.000 | 0.003 | **0.062** | 0.000 / 0.000 | d6 ("Original"), `d6_run5.txt` |
| A3 | Run 4, época 11 (`794a049:archivo/best_m5.keras`, SHA `bb30e9fd…cbba8`) | original | no / no | 0.5 | 0.5 | 0.001 | 0.179 | 0.099 | 0.000 | 0.003 | **0.056** | 0.000 / 0.000 | `evaluate.py` directo (salida pegada por el usuario; no hay archivo en el repo) |
| A4 | Run 4, época 11, **re-guardado** antes de evaluar | original | no / no | 0.5 | 0.5 | 0.001 | 0.207 | 0.114 | 0.000 | 0.003 | **0.065** | 0.000 / 0.000 | d6, `d6_run4.txt` |
| **Grupo B: BN recalibrado (momentum 0.9, 3 pasadas, train sin aumentos; pesos intactos; en memoria y re-guardado)** | | | | | | | | | | | | | |
| B1 | Run 5, época 8 | recalibrado | sí / sí | 0.5 | 0.5 | 0.053 | 0.519 | 0.294 | 0.334 | 0.216 | **0.283** | 1.000 / 0.008 | d6, `d6_run5.txt` |
| B2 | ídem B1 | recalibrado | sí / sí | 0.2 | 0.5 | (= B1, AP no depende del score) | | | | | **0.283** | 0.175 / 0.381 | d6 (solo se imprimieron P, R y mAP) |
| B3 | Run 4, época 11 | recalibrado | no / no | 0.5 | 0.5 | 0.188 | 0.683 | 0.389 | 0.032 | 0.266 | **0.311** | 0.755 / 0.059 | d6, `d6_run4.txt` |
| B4 | ídem B3 | recalibrado | no / no | 0.2 | 0.5 | (= B3) | | | | | **0.311** | 0.276 / 0.345 | d6 |
| B5 | Run 5 derivado `archivo/best_m5_bnrecal.keras` (SHA `b6631424…15a3e`), generado con `recalibrar_bn.py` | recalibrado (E6a) | sí / sí | 0.5 | 0.5 | 0.059 | 0.525 | 0.290 | 0.329 | 0.212 | **0.283** | 1.000 / 0.006 | `evaluate.py` directo (`archivo/evaluate_m5_bnrecal.txt`) |
| (d7) | Run 5 original, 5 variantes en un proceso | original | sí / sí | 0.5 | 0.5 | 0.001 | 0.191 | 0.102 | 0.000 | 0.003 | 0.059 | 0.000 / 0.000 | `d7_run5.txt` (grupo A; tercer valor del mismo checkpoint) |

*Histórico, sin condiciones registradas [bitácora; no reproducible desde el repo]:* mAP de los Runs 1–4: 0.000, 0.065, 0.061, 0.055 (todos sin EXIF y con BN original). No se mezclan con el resto. Tampoco se usa el mAP de `d3` (0.057 / 0.065): `d3` reordena empates con un `argsort` no estable.

**14.10.2 Comparación normalizada de Run 4 y Run 5 (qué se puede y qué no se puede afirmar)**
- **Efecto de BN (limpio):** mismo checkpoint, mismas condiciones de evaluación, solo cambian las estadísticas móviles: A→B, Run 5 0.055–0.062 → **0.283** y Run 4 0.056–0.065 → **0.311**. El aumento (≈ +0.22/+0.25) es unas 25 veces el ruido de medida de ≈ 0.01. Es el único efecto que los datos aíslan.
- **Run 4 frente a Run 5 (confundido):** con BN recalibrado y umbrales idénticos (B1 frente a B3), mAP 0.283 frente a 0.311 y moneda 0.334 frente a 0.032. **No se atribuye a EXIF ni a BN**, porque difieren a la vez: (1) EXIF en el **entrenamiento** y en la **evaluación** (el Run 4 se evaluó sobre imágenes de val sin orientar con etiquetas del marco orientado: en las 247 imágenes con EXIF 3/6/8 las GT no caen sobre el objeto; parte del AP bajo de moneda del Run 4 puede ser un artefacto de la evaluación y no del modelo); (2) la época (8 frente a 11); (3) el run (semilla de pesos y orden de datos distintos; un solo run por condición; no se conoce la varianza entre runs). El BN no es un factor de diferencia entre B1 y B3, porque se recalibraron con el mismo procedimiento.
- **Ruido de medida:** A1 frente a A2 y A3 frente a A4 difieren en 0.007–0.009 con el mismo checkpoint (ver 14.10.4).
- **Evaluación cruzada propuesta (sin entrenar):** recalibrar cada checkpoint con la bandera con la que se entrenó y evaluarlo con **las dos** banderas (Run 4 con EXIF 0 y 1; Run 5 con EXIF 1 y 0). Separa el efecto del marco de evaluación del efecto del entrenamiento con EXIF en moneda y laptop. Pendiente: los comandos se piden al usuario (ver el informe de la sesión).

**14.10.3 Herramienta `recalibrar_bn.py` y pruebas (`tests/test_recalibrar_bn.py`, 16 pruebas, `.venv` de Windows, CPU, ≈ 70 s, todas OK)**
- **Qué hace:** carga el checkpoint, hace `training=True` hacia adelante sobre las imágenes de `data/train` (sin aumentos, misma carga que el entrenamiento, incluida la orientación EXIF si `APLICAR_EXIF=1`; orden aleatorio fijado por `semilla + nº de pasada`), con momentum 0.9 (3 pasadas por defecto), restaura el momentum original de cada capa y guarda un `.keras` derivado **nuevo**; escribe `<derivado>.recal.json` con SHA256 de origen y derivado, bandera EXIF, parámetros y versiones. Antes de guardar comprueba que ningún peso distinto de `moving_mean`/`moving_variance` cambió y que alguna estadística cambió; tras guardar relee el derivado y lo compara con el modelo en memoria, y comprueba que el SHA256 del original no cambió. Nunca sobrescribe un archivo existente ni acepta `--salida` igual a `--modelo`. No hay optimizador ni gradientes.
- **Pruebas y qué verifican:** (1) pesos entrenables, gamma y beta idénticos bit a bit; (2) solo cambian las estadísticas móviles (las 116 de las 58 capas BN cambian; la función `comparar` detecta un peso modificado); (3) el momentum original se conserva; (4) el SHA256 del original no cambia, no se sobrescribe, error con el mismo archivo / inexistente; (5) reproducible con la misma semilla (idéntico) y distinto con otra semilla; (6) solo se usan rutas de `data/train` (ninguna de val, comprobado interceptando `dataset_solo_imagenes`); (7) EXIF: con 16 imágenes de train con EXIF 3/6/8 la bandera cambia las estadísticas obtenidas y es reproducible con la misma bandera, y el `.recal.json` registra la bandera activa; (8) consistencia: con BN de lote (`training=True`) original y derivado dan la misma salida (`allclose` 1e-5), con BN de inferencia difieren, y guardar/recargar el derivado no cambia las predicciones (igualdad exacta); (9) `evaluate.predecir` es determinista (misma salida dos veces) y no toca el checkpoint; (10) la línea de comandos funciona de extremo a extremo y se niega a sobrescribir. **Límites:** los tests usan un modelo de pesos aleatorios (con estadísticas BN perturbadas a propósito), CPU de Windows y 1–2 lotes; no prueban la calidad del resultado con un checkpoint real ni el comportamiento en GPU/WSL2 (eso es la ejecución del usuario).
- **Ejecutada sobre el Run 5 real (usuario, WSL2, `APLICAR_EXIF=1`):** `archivo/best_m5_bnrecal.keras` (SHA256 `b6631424…15a3e`) + `archivo/best_m5_bnrecal.recal.json`. Informe: original sin cambios (SHA256 `0f092594…93526`), 1097 imágenes de train, 3 pasadas, 207 pasos, 323 pesos, 116 estadísticas de BN, **116 cambiadas y 0 pesos no-estadística cambiados**. `evaluate.py` sobre el derivado (score 0.5, IoU 0.5, EXIF 1; `archivo/evaluate_m5_bnrecal.txt`): bolígrafo 0.059, laptop 0.525, libro 0.290, moneda 0.329, mouse 0.212, **mAP 0.283**, P 1.000, R 0.006. Fila **B5** de la tabla (grupo B). **Reproduce el mAP de d6 (B1: 0.283)**; por clase las diferencias con B1 son ≤ 0.006 (0.053/0.519/0.294/0.334/0.216), del orden del ruido entre procesos de 14.10.4. Con el Run 4 aún no se ha ejecutado la herramienta. Nota: al inicio de `evaluate.py` la GPU registró errores `CUDA_ERROR_OUT_OF_MEMORY` al reservar bloques de ~5 GB (RTX 4060, 5.5 GB utilizables); TF continuó y la evaluación terminó (aviso sin efecto en los resultados).

**14.10.4 Discrepancia entre `evaluate.py` directo (A1, A3) y el "Original" de d6 (A2, A4)**
- **Hechos:** el mismo checkpoint da mAP 0.055 → 0.062 (Run 5) y 0.056 → 0.065 (Run 4). Los dos checkpoints cambian **en el mismo sentido y solo en laptop y libro** (laptop +0.026/+0.028, libro +0.008/+0.015; bolígrafo, moneda y mouse idénticos). Mismas condiciones nominales (score 0.5, IoU 0.5, misma bandera EXIF). La diferencia entre las dos rutas es que d6 hace cargar → `modelo.save(tmp)` → `evaluar(tmp)` y lo ejecuta tras otras pasadas con otros modelos en el mismo proceso.
- **Resultado de `d7` (Run 5, WSL2, GPU, `APLICAR_EXIF=1`; `diagnostico/run5/d7_run5.txt`):** las cinco variantes dan **exactamente lo mismo** (mAP 0.059; bolígrafo 0.001, laptop 0.191, libro 0.102, moneda 0.000, mouse 0.003): evaluar el original dos veces, evaluar la copia re-guardada dos veces y evaluar tras actividad `training=True` de otro modelo. Los 323 pesos del original y de la copia son idénticos (diferencia máxima 0), el momentum de BN coincide, y las salidas crudas de `predict` (cajas y scores) son **idénticas bit a bit** entre repetición y entre original y copia (2,773,806 scores > 0.01 en ambos). El score máximo de clase en val es 0.411 (< 0.5, de ahí precisión y recall 0 con score 0.5).
- **Conclusión (hecho + inferencia):** re-guardar el modelo y la actividad previa en el proceso **no** causan la diferencia (descartado en GPU y en CPU). Dentro de un mismo proceso el resultado es exactamente reproducible, pero **entre procesos distintos** el mismo checkpoint dio **tres** valores: 0.055 (A1), 0.059 (d7) y 0.062 (A2); laptop 0.163 / 0.191 / 0.189 y libro 0.109 / 0.102 / 0.117. El "mismo sentido" de A2 y A4 frente a A1 y A3 era, con este tercer valor, compatible con casualidad. La explicación más consistente (**no demostrada**: requeriría varias ejecuciones independientes de la misma orden) es variación numérica entre procesos de GPU/cuDNN/XLA (autotuning de algoritmos y oneDNN, ya avisada por TF en el log), amplificada porque con BN original los scores son bajos y casi empatados y el AP depende de detecciones de score 0.01–0.05. **Ruido entre procesos con BN original: ≈ ±0.004 de mAP (rango 0.007) y hasta ±0.014 en el AP de una clase; usar esas cotas, no 0.01 genérico.** Con BN recalibrado la medición es más estable (B1 y B5: mismo mAP 0.283 en dos procesos distintos).
- **Discrepancia: resuelta como ruido entre procesos, con la reserva anterior.** Implicación: las diferencias de mAP < ≈ 0.01 con BN original (p. ej. 0.055 frente a 0.056) no son interpretables.
- **Descartado en CPU (Windows):** guardar y recargar un modelo no cambia sus predicciones (igualdad exacta) y `evaluate.predecir` es determinista (pruebas 8 y 9 de arriba, con un modelo aleatorio); `d7` lo confirma en la GPU de WSL2.
- **Hipótesis (sin probar):** (a) variación numérica de GPU/cuDNN entre ejecuciones, que con BN desfasado (scores diminutos y casi empatados, AP dominado por detecciones de score 0.01–0.05) reordena detecciones; la dirección común sería casualidad o consecuencia de esa sensibilidad; (b) un efecto de la re-guardada específico del entorno (versiones TF 2.22.0rc1 / keras-nightly). El comportamiento sistemático (mismo sentido en dos checkpoints) hace menos probable el ruido puro; **no resuelto**.
- **Diagnóstico (`diagnostico/run5/d7_discrepancia.py`, solo inferencia):** en un solo proceso evalúa el mismo checkpoint dos veces, la copia re-guardada dos veces, y de nuevo tras actividad `training=True`; compara pesos (original frente a copia) y salidas crudas de `predict`. Ejecutado por el usuario sobre el Run 5 (resultado arriba). Pendiente opcional: ejecutarlo sobre el Run 4.

**14.10.5 Evaluación cruzada EXIF con BN recalibrado (ejecutada por el usuario en WSL2; `diagnostico/run5/e6a_cruzada.txt`)**
Cada checkpoint se recalibró con la bandera con la que se entrenó (`recalibrar_bn.py`, 3 pasadas, momentum 0.9, semilla 0; 116/116 estadísticas cambiadas, 0 pesos más; SHA256 del Run 4 verificado) y se evaluó con las dos banderas. Score 0.5, IoU 0.5 en todo. Los derivados del Run 4 están en `/tmp` (no versionados).

| Checkpoint recalibrado (entrenamiento) | Evaluación | bolígrafo | laptop | libro | moneda | mouse | **mAP** | P / R |
|---|---|---|---|---|---|---|---|---|
| Run 4, época 11 (sin EXIF) | EXIF=0 | 0.190 | 0.651 | 0.369 | 0.033 | 0.257 | **0.300** | 0.700 / 0.056 |
| Run 4, época 11 (sin EXIF) | EXIF=1 | 0.189 | 0.653 | 0.339 | 0.056 | 0.250 | **0.297** | 0.708 / 0.054 |
| Run 5, época 8 (EXIF) | EXIF=1 | 0.059 | 0.525 | 0.290 | 0.329 | 0.212 | **0.283** | 1.000 / 0.006 |
| Run 5, época 8 (EXIF) | EXIF=0 | 0.059 | 0.521 | 0.291 | 0.076 | 0.208 | **0.231** | 1.000 / 0.006 |

- **Hechos:**
  1. **Marco de evaluación, Run 4:** pasar de EXIF=0 a EXIF=1 cambia la moneda de 0.033 a 0.056 (+0.023) y el mAP en −0.003. Bolígrafo, laptop y mouse no cambian; libro cambia 0.030 **sin tener ninguna imagen con EXIF** (ver ruido abajo). Es decir, evaluar el Run 4 en el marco con las etiquetas correctas **no rescata la moneda**: el fallo del Run 4 en moneda no es un artefacto de la evaluación.
  2. **Marco de evaluación, Run 5:** pasar de EXIF=1 a EXIF=0 hunde la moneda de 0.329 a 0.076 (−0.253) y el mAP de 0.283 a 0.231. Laptop, libro, bolígrafo y mouse no cambian (±0.004). Un modelo entrenado con imágenes orientadas necesita recibirlas orientadas.
  3. **Comparación en el mismo marco correcto (EXIF=1 para ambos):** moneda **0.329 (Run 5) frente a 0.056 (Run 4)**, una diferencia de 0.27, muy por encima del ruido. En cambio el Run 4 es mejor en bolígrafo (0.189 frente a 0.059), laptop (0.653 frente a 0.525), libro (0.339 frente a 0.290) y mouse (0.250 frente a 0.212), de modo que el mAP total es parecido (0.297 frente a 0.283). Bolígrafo, libro y mouse no tienen imágenes con EXIF: esas diferencias **no** vienen de la orientación.
- **Lectura (separando hechos de inferencia):** (a) el entrenamiento con EXIF es lo que permite aprender la moneda: la mejora de 0.056 a 0.329 aparece con el marco de evaluación fijado y no se explica por él. (b) Las pérdidas del Run 5 en las clases sin EXIF indican otro factor (época 8 frente a 11, semilla/run, o un efecto indirecto del EXIF) **no aislado**: solo hay un run por condición y checkpoints de épocas distintas. (c) La conclusión de §13.3 ("el EXIF no era la causa principal") queda **refutada** para moneda, no para el mAP global.
- **Implicación para la app (hipótesis a verificar, no probada):** el modelo entrenado con EXIF rinde mal con imágenes sin orientar (moneda 0.329 → 0.076), así que la app deberá entregar al modelo imágenes con la orientación aplicada, como `APLICAR_EXIF=1`; hay que decidirlo antes de exportar (la bandera no viaja con el modelo).
- **Ruido y variabilidad:** libro (sin EXIF) varió 0.030 entre dos evaluaciones del mismo Run 4 recalibrado (0.369 frente a 0.339); por tanto la cota de ±0.014 por clase de 14.10.4 queda **revisada a ≈ ±0.03 por clase** (mAP: diferencias de ≈ 0.003–0.007 entre procesos). Además, la recalibración del Run 4 con la herramienta dio mAP **0.300** frente a **0.311** de d6 (laptop 0.651 frente a 0.683, libro 0.369 frente a 0.389), mientras que en el Run 5 coinciden (0.283 y 0.283). Diferencia posible (hipótesis, no probada): d6 y la herramienta recorren las imágenes de train en distinto orden y con el momentum 0.9 los últimos lotes pesan más. Se puede medir la variabilidad de la propia recalibración repitiéndola con otras semillas (`--semilla 1`, `--semilla 2`); medido en §14.11.

### 14.11 Fase 1: variabilidad de la recalibración de BN (semillas 0, 1 y 2; ejecutada por el usuario en WSL2; `diagnostico/run5/variabilidad_recal.txt`)
**Configuración (constante):** `recalibrar_bn.py` con 3 pasadas, momentum 0.9, solo train (1097 imágenes, 207 pasos de lote); `evaluate.py` con score 0.5 e IoU 0.5; cada checkpoint en el marco EXIF de su entrenamiento (Run 4 `APLICAR_EXIF=0`, Run 5 `APLICAR_EXIF=1`). Solo cambia `--semilla`. Orígenes verificados por SHA256: **Run 4** `/tmp/run4.keras` = `git show 794a049:archivo/best_m5.keras`, `bb30e9fd…cbba8`; **Run 5** `archivo/best_m5.keras`, `0f092594…93526` (el nombre `best_m5.keras` designa archivos distintos en `794a049` y en `HEAD`). Todas las recalibraciones: 116/116 estadísticas cambiadas, 0 pesos más, original sin cambios. SHA256 de los seis derivados en el `.txt`. Los derivados de `/tmp` no están versionados.

| Checkpoint (EXIF) | Semilla | bolígrafo | laptop | libro | moneda | mouse | **mAP** | P / R |
|---|---|---|---|---|---|---|---|---|
| Run 4 (0) | 0 | 0.190 | 0.651 | 0.369 | 0.033 | 0.257 | **0.300** | 0.700 / 0.056 |
| Run 4 (0) | 1 | 0.187 | 0.640 | 0.365 | 0.033 | 0.260 | **0.297** | 0.750 / 0.058 |
| Run 4 (0) | 2 | 0.189 | 0.650 | 0.373 | 0.033 | 0.266 | **0.302** | 0.708 / 0.054 |
| Run 5 (1) | 0 | 0.059 | 0.525 | 0.290 | 0.329 | 0.212 | **0.283** | 1.000 / 0.006 |
| Run 5 (1) | 1 | 0.057 | 0.522 | 0.289 | 0.339 | 0.209 | **0.283** | 1.000 / 0.010 |
| Run 5 (1) | 2 | 0.057 | 0.515 | 0.289 | 0.329 | 0.210 | **0.280** | 1.000 / 0.010 |

| | bolígrafo | laptop | libro | moneda | mouse | mAP |
|---|---|---|---|---|---|---|
| Run 4: media (rango) | 0.189 (0.003) | 0.647 (0.011) | 0.369 (0.008) | 0.033 (0.000) | 0.261 (0.009) | **0.300 (0.005)** |
| Run 5: media (rango) | 0.058 (0.002) | 0.521 (0.010) | 0.289 (0.001) | 0.332 (0.010) | 0.210 (0.003) | **0.282 (0.003)** |
| Diferencia Run 4 − Run 5 | +0.131 | +0.126 | +0.080 | −0.299 | +0.051 | +0.018 |

- **Hechos:** (1) Con 3 semillas, la variabilidad atribuible a la recalibración (más la evaluación) es **pequeña**: rango del mAP 0.005 (Run 4) y 0.003 (Run 5); por clase, rango ≤ 0.011. (2) Ninguna medida de un checkpoint solapa con las del otro en ninguna clase ni en el mAP (mín(Run 4) − máx(Run 5): bolígrafo +0.128, laptop +0.115, libro +0.075, moneda −0.306, mouse +0.045, mAP +0.014). (3) La diferencia de medias entre checkpoints supera al mayor de los dos rangos (el de cualquiera de los checkpoints) en: bolígrafo ≈ 44 veces, moneda ≈ 30, laptop ≈ 11, libro ≈ 10, mouse ≈ 6 (0.051 frente a 0.009) y el mAP ≈ 3.6 veces (0.018 frente a 0.005).
- **Qué significa y qué no:** esas diferencias son **consistentes frente a la variabilidad de recalibración y de evaluación** medida; son diferencias reales entre estos dos checkpoints. **No** miden la variabilidad entre entrenamientos (un solo run por condición): no se puede afirmar que otro run de la misma configuración daría lo mismo. La diferencia de mAP total (0.018) es la menos holgada; las de clase son grandes.
- **Matices:** el Run 4 se evaluó en su marco (EXIF=0) y el Run 5 en el suyo (EXIF=1); con EXIF=1 el Run 4 había dado mAP 0.297 y moneda 0.056 (§14.10.5), así que usar el mismo marco para ambos no altera el orden. La discrepancia 0.300 (herramienta) frente a 0.311 (d6) del Run 4 queda **dentro de lo atribuible al orden de lotes**: con las tres semillas de la herramienta el máximo es 0.302; **0.311 queda fuera** (> 0.009 por encima), compatible con la hipótesis del distinto orden/mezcla de d6, no probada. En cuanto al libro del Run 4 (0.339 con EXIF=1 frente a 0.365–0.373 con EXIF=0), los seis valores de la tabla en condiciones iguales varían ≤ 0.008, así que aquel 0.339 sigue **sin explicación** (la bandera EXIF no cambia ninguna imagen de libro).
- **Cota de ruido para decidir (provisional):** para checkpoints recalibrados con este procedimiento, tratar como no distinguibles las diferencias de mAP < ≈ 0.01 y de AP por clase < ≈ 0.02 (3 muestras por checkpoint; la cota por clase sigue sujeta a la incertidumbre de n=3 y al valor aislado de libro).

### 14.12 Preparación de E6b: momentum de BatchNorm del backbone 0.999 → 0.99 (24-nov-2026; autorizado preparar, **no entrenar**)
**Estado:** cambios y tests hechos; **ningún entrenamiento ejecutado por Claude**. Los dos entrenamientos (E6b-EXIF = "Run 6", E6b-Legacy = "Run 7") los lanza el usuario en WSL2. Cuando haya resultados se registrarán aquí y en CLAUDE.md. Hipótesis a contrastar (no conclusión): con momentum 0.99 las estadísticas móviles convergen en ≈ 1 época (0.99^69 ≈ 0.5), la `val_loss` y la selección del mejor checkpoint serán fiables y el modelo en inferencia rendirá cerca de lo que hoy da la recalibración (≈ 0.28–0.30 de mAP) sin necesidad de recalibrar.

**Cambios en el código (mínimos y reversibles)**
- `config.py`: `BN_MOMENTUM_BACKBONE`, leído de la variable de entorno `BN_MOMENTUM_BACKBONE`; `None` si no se define (= 0.999 de Keras, **comportamiento legacy y de los Runs 1–5, sin cambios por defecto**); valor fuera de (0, 1) → `AssertionError`. Mismo patrón que `APLICAR_EXIF`.
- `model.py` (`construir_ssd`): si la constante no es `None`, asigna ese momentum **solo** a las `BatchNormalization` de `keras.applications.MobileNetV2` (52 capas). El valor viaja dentro del `.keras` (probado: guardar/recargar lo conserva), así que `evaluate.py`, `recalibrar_bn.py` y el export no necesitan la variable.
- **Dependencia con otras capas (verificado en el código y en los tests):** los 6 BN de los 3 bloques extra se crean con `layers.BatchNormalization(name=…)` sin momentum, es decir, con el valor por defecto de Keras **0.99**; el cambio no los afecta y no se tocan. Con E6b las 58 capas BN quedan en 0.99.
- `train.py`: (1) opción **`--semilla N`** (por defecto `None` = no se fija ninguna semilla, como antes), que llama a `tf.keras.utils.set_random_seed(N)`; (2) imprime al inicio la configuración (`APLICAR_EXIF`, `BN_MOMENTUM_BACKBONE`, semilla, batch, lr). No se tocó ningún hiperparámetro, callback ni dato. **Aviso:** esta opción no estaba en la lista literal de cambios autorizados; se añadió porque la comparación EXIF/legacy pidió "las mismas semillas" y no existía forma de fijarlas. Es opcional y revertible.
- **Revertir E6b:** no definir `BN_MOMENTUM_BACKBONE` (nada más); para quitar el código, eliminar el bloque `if C.BN_MOMENTUM_BACKBONE …` de `model.py`, la constante de `config.py` y la opción `--semilla` de `train.py`.

**Qué garantiza `--semilla` y qué no (`diagnostico/run5/d8_semilla_pipeline.py`, `d8_semilla_pipeline.txt`, ejecutado en Windows/CPU)**
- **Garantizado (verificado):** (a) los **pesos iniciales** son idénticos con la misma semilla, con cualquier valor de `APLICAR_EXIF` y con o sin el cambio de momentum (huella SHA256 de los 323 pesos `4df2ef18…6bf2` con semilla 42 en los tres casos; `diagnostico/e6b_verificar_config.py`); (b) el **orden de barajado** de las imágenes por época es el mismo con la misma semilla y distinto con otra.
- **No garantizado (verificado):** (c) los **aumentos aleatorios** (flip, zoom, brillo, contraste) en el `map` paralelo (AUTOTUNE) **no** se reproducen: con la misma semilla solo 2–6 de 16 imágenes aumentadas coincidieron entre ejecuciones; y las GPU introducen variación numérica (ya medida en §14.10.4). Por tanto E6b-EXIF y E6b-Legacy partirán del mismo estado y verán las imágenes en el mismo orden, pero **no serán idénticos salvo por EXIF**: la variabilidad entre entrenamientos no se conoce (un run por condición) y limita lo que se podrá concluir. Cambiar `num_parallel_calls` para volverlo reproducible alteraría el pipeline y **no** se hizo (fuera de lo autorizado).

**Tests y verificaciones (Windows, CPU; no entrenan)**
- `tests/test_bn_momentum.py` (12 tests, OK): por defecto backbone 0.999 y extras 0.99; con 0.99 → 52 + 6 = 58 capas, todas 0.99; ningún otro atributo de BN cambia (nombre, epsilon, eje, center, scale); misma forma y número de pesos y **misma salida en inferencia** tras copiar los pesos; el momentum **actúa** en la actualización (tras un paso `training=True` desde las mismas estadísticas iniciales, `moving_mean` con 0.99 es 10× la de 0.999 y `moving_variance − 1` también); el valor se guarda y recarga en el `.keras`; lectura de la variable de entorno (ausente, vacía, `0.99`, y rechazo de `1.5`, `0`, `-0.1`, `1`); misma semilla → mismos pesos iniciales (con y sin cambio de momentum); inspección estática (`ast`) de que `--semilla` existe en `train.py` con valor por defecto `None` (train.py **no** se importa ni se ejecuta).
- Suites completas tras el cambio: `test_exif` (15) + `test_recalibrar_bn` (16) + `test_bn_momentum` (12) = **43 tests OK** (≈ 90 s).
- **Verificado en WSL2 por el usuario (Python 3.14, TF 2.22.0-rc1, Keras 3.16.0.dev2026111903):** `diagnostico/e6b_verificar_config.py` con `APLICAR_EXIF=1 BN_MOMENTUM_BACKBONE=0.99`: 52 BN del backbone y 6 extras, todas en 0.99; huella SHA256 de los 323 pesos iniciales con semilla 42 `4df2ef18…6bf2`, **idéntica a la de Windows**. `python -B -m unittest tests.test_bn_momentum -v`: **12/12 OK** (12.8 s), incluido el efecto real del momentum en la actualización y su persistencia en el `.keras`.

**Plan de ejecución (a cargo del usuario; una orden por vez)**
1. Verificar configuración y tests en WSL2 (sin entrenar).
2. **Run 6 = E6b-EXIF:** `APLICAR_EXIF=1 BN_MOMENTUM_BACKBONE=0.99 python train.py --epocas 150 --semilla 42`; luego archivar `salida/best_model.keras` → `archivo/best_m6.keras` y `salida/historial.csv` → `archivo/historial_m6.csv`, y evaluar con la misma bandera.
3. **Run 7 = E6b-Legacy:** igual con `APLICAR_EXIF=0` (misma semilla, mismo momentum) → `best_m7.keras`, `historial_m7.csv`.
4. Evaluar cada modelo **sin recalibrar** y, como control, recalibrado (para ver si BN ya convergió: una recalibración que cambie poco el mAP lo confirmaría), y en los dos marcos EXIF (cruzada 2×2 como en §14.10.5). Condiciones fijas: score 0.5, IoU 0.5.
5. Comparación final: Run 4 recalibrado, Run 5 recalibrado, Run 6, Run 7. No se elige modelo para la app todavía.
`train.py` sobrescribe `salida/best_model.keras` e `historial.csv`: el Run 5 ya está archivado (`archivo/best_m5.keras`, SHA `0f092594…93526`), pero hay que archivar el Run 6 **antes** de lanzar el Run 7.

### 14.13 Run 6 = E6b-EXIF: resultados (entrenado y evaluado por el usuario en WSL2, 24/25-nov-2026)
**Configuración (impresa por `train.py`):** `APLICAR_EXIF=1 | BN_MOMENTUM_BACKBONE=0.99 | semilla=42 | batch=16 | lr=0.001`, GPU detectada. Comando: `APLICAR_EXIF=1 BN_MOMENTUM_BACKBONE=0.99 python train.py --epocas 150 --semilla 42`. Todo lo demás igual que el Run 5 (arquitectura, anchors, zoom, dataset y reparto, callbacks, paciencia 30).

**Archivos:** `archivo/best_m6.keras` (SHA256 `24b06622f628b0a791b606072e74ea88185f8712d8ebbb1c3201708fb1b99d73`, idéntico a `salida/best_model.keras`), `archivo/historial_m6.csv` (108 épocas, idéntico a `salida/historial.csv`), `archivo/evaluate_m6.txt`, `archivo/best_m6_bnrecal.keras` (SHA256 `a8517b6f760d96b2005bd91d141bf9670e5ca83d12abad564e85294222239258`) + `.recal.json`, `archivo/evaluate_m6_bnrecal.txt`, `diagnostico/run5/e6b_cruzada.txt`. Hashes comprobados por Claude desde Windows.

**Entrenamiento (`historial_m6.csv`):**
- Mejor `val_loss` **1.9233 en la época 78**; parada por `EarlyStopping` en la época 108 (78 + 30). Run 5 (mismo marco EXIF, misma pérdida, mismo val): mejor 7.9220 en la época 8, parada en la 38.
- La `val_loss` baja de forma sostenida (10.01 → 5.11 en la época 9 → 2.86 en la 25 → 2.08 en la 50 → 1.92 en la 78) y luego queda plana (1.93–1.94); no hay la divergencia del Run 5 (que subía a 14.41). `loss` final 1.33–1.38: brecha train–val ≈ 0.55, pequeña.
- LR: 1e-3 hasta la época 24; `ReduceLROnPlateau` lo baja a 5e-4 (25), 2.5e-4 (46), 1.25e-4 (56), 6.25e-5 (66), 3.1e-5 (71) y hasta 1e-6 (104). Cada reducción coincide con una bajada clara de la `val_loss` en las primeras.

**Evaluación (val, score 0.5 para precisión/recall, IoU 0.5; AP independiente del score):**

| AP@0.5 | Run 4 recal.¹ | Run 5 recal.¹ | **Run 6** | Run 6 recal. | Run 6 en marco EXIF=0² |
|---|---|---|---|---|---|
| bolígrafo | 0.189 | 0.058 | **0.552** | 0.533 | 0.552 |
| laptop | 0.647 | 0.521 | **0.995** | 0.991 | 0.799 |
| libro | 0.369 | 0.289 | **0.901** | 0.909 | 0.901 |
| moneda | 0.033 | 0.332 | **0.979** | 0.980 | 0.152 |
| mouse | 0.261 | 0.210 | **0.956** | 0.923 | 0.953 |
| **mAP@0.5** | 0.300 | 0.282 | **0.877** | 0.867 | 0.671 |
| Precisión / recall | — | 1.000 / 0.006³ | **0.921 / 0.657** | 0.902 / 0.635 | 0.708 / 0.502 |

¹ Medias de las semillas 0/1/2 (§14.11). ² No es métrica válida del modelo (ver abajo). ³ Semilla 0 (`evaluate_m5_bnrecal.txt`).

**Hechos:**
- El Run 6 **sin recalibrar** alcanza mAP@0.5 = **0.877**, por encima de la meta de 0.50; todas las clases suben, y la diferencia con los Runs 4/5 recalibrados (≈ +0.58 de mAP) supera en más de 50× la variabilidad medida de la evaluación (< 0.01).
- **Recalibrar ya no mejora** (0.877 → 0.867, −0.010, en el límite del ruido; por clase hasta −0.033 en mouse, dentro del ruido ±0.03 de evaluaciones aisladas): con momentum 0.99 las estadísticas de BN quedan convergidas al terminar el entrenamiento; para este run se usa el checkpoint original.
- **Cruzada EXIF:** con `APLICAR_EXIF=0`, bolígrafo y libro (clases sin EXIF 3/6/8) dan exactamente el mismo AP que con `=1` (también en procesos distintos) y mouse varía 0.003; caen laptop (0.995 → 0.799) y moneda (0.979 → 0.152), las clases con imágenes de orientación 3/6/8. Esa evaluación no es válida como métrica (las cajas de val de esas imágenes están en el marco orientado y los píxeles no), pero indica que **el modelo debe recibir imágenes orientadas en inferencia** (implicación para la app, por verificar).

**Hipótesis (coherente con los datos, no demostrada):** en los Runs 1–5 la `val_loss` en modo inferencia estaba falseada por las estadísticas de BN desfasadas (momentum 0.999); `ModelCheckpoint`, `ReduceLROnPlateau` y `EarlyStopping` decidían con esa señal y elegían las épocas 8–11 y paraban cuando el modelo seguía aprendiendo. Eso explicaría que el salto del Run 6 (0.28 → 0.88) sea mucho mayor que el de recalibrar un checkpoint temprano (0.06 → 0.28). No está aislado cuánto aporta cada parte (BN correcto en sí frente a entrenamiento más largo con LR decreciente).

**Limitaciones:** un solo run (la variabilidad entre entrenamientos sigue sin conocerse; `--semilla` no reproduce los aumentos ni la GPU); no hay conjunto de prueba independiente y val se usó para elegir el checkpoint y para el LR (el 0.877 es algo optimista); bolígrafo sigue siendo la clase débil (0.552), coherente con su cobertura de anchors (§14). **No se elige modelo para la app** hasta comparar con el Run 7 (E6b-Legacy).

### 14.14 Run 7 = E6b-Legacy y comparación final E6b (entrenado y evaluado por el usuario en WSL2, 25-nov-2026)
**Configuración (impresa por `train.py`):** `APLICAR_EXIF=0 | BN_MOMENTUM_BACKBONE=0.99 | semilla=42 | batch=16 | lr=0.001`, GPU. Comando: `APLICAR_EXIF=0 BN_MOMENTUM_BACKBONE=0.99 python train.py --epocas 150 --semilla 42`. Idéntico al Run 6 salvo `APLICAR_EXIF` (y la variación no controlable de aumentos y GPU, §14.12).

**Archivos:** `archivo/best_m7.keras` (SHA256 `86713516ebb688edda4d7eebea2236210db6a29fbc5417692385251727538375b`), `archivo/historial_m7.csv` (143 épocas), `archivo/evaluate_m7.txt` (marco EXIF=0), `archivo/best_m7_bnrecal.keras` (SHA256 `be80c8705366fee1972f2ffd50b22c1829dcd140861b0feb3282a6cbfbae2de9`) + `.recal.json` (recalibrado con `APLICAR_EXIF=0`), `archivo/evaluate_m7_bnrecal.txt`, `diagnostico/run5/e6b_cruzada.txt` (C1–C4). Hashes comprobados por Claude desde Windows (copias idénticas a `salida/`; `best_m6.keras` intacto).

**Entrenamiento (`historial_m7.csv`):** mejor `val_loss` **2.6696 en la época 113**, parada en la 143 (113 + 30); baja sostenida sin divergencia, como el Run 6; LR 1e-3 hasta la época 33 y luego reducciones hasta 1e-6 (época 128). `loss` final ≈ 2.1–2.2 (Run 6: ≈ 1.4). La `val_loss` **no es comparable** con la del Run 6: con EXIF=0, 247 imágenes de train y val tienen cajas desalineadas con los píxeles, así que la pérdida mide otra cosa (hipótesis: esa contradicción explica la `loss` más alta).

**Tabla final (val, score 0.5 para precisión/recall, IoU 0.5; AP independiente del score). Marco de evaluación indicado; el marco correcto (cajas alineadas con los píxeles) es EXIF=1:**

| AP@0.5 | Run 4 recal.¹ (E=0) | Run 5 recal.¹ (E=1) | **Run 6** (E=1) | Run 6 recal. (E=1) | Run 7 (E=0)² | **Run 7** (E=1) | Run 7 recal. (E=1) |
|---|---|---|---|---|---|---|---|
| bolígrafo | 0.189 | 0.058 | **0.552** | 0.533 | 0.517 | **0.517** | 0.502 |
| laptop | 0.647 | 0.521 | **0.995** | 0.991 | 0.981 | **0.982** | 0.978 |
| libro | 0.369 | 0.289 | **0.901** | 0.909 | 0.849 | **0.848** | 0.847 |
| moneda | 0.033 | 0.332 | **0.979** | 0.980 | 0.382 | **0.809** | 0.772 |
| mouse | 0.261 | 0.210 | **0.956** | 0.923 | 0.912 | **0.896** | 0.868 |
| **mAP@0.5** | 0.300 | 0.282 | **0.877** | 0.867 | 0.728 | **0.810** | 0.793 |
| Precisión / recall | — | — | 0.921 / 0.657 | 0.902 / 0.635 | 0.870 / 0.450 | 0.890 / 0.495 | 0.875 / 0.484 |

¹ Medias de las semillas 0/1/2 (§14.11), cada uno en su marco de entrenamiento; Run 4 en EXIF=1: 0.297 (§14.10.5). ² Marco de entrenamiento del Run 7; no válido para moneda/laptop (cajas desalineadas).

**Hechos:**
- **BN es el factor dominante.** Con momentum 0.99 ambos runs superan con amplitud la meta de 0.50: Run 7 (misma configuración que el Run 4 salvo el momentum, y época elegida por la `val_loss`) **0.810** frente a ≈ 0.30 del Run 4 recalibrado; Run 6 **0.877** frente a 0.282 del Run 5 recalibrado.
- **Recalibrar ya no aporta con momentum 0.99:** Run 6 0.877 → 0.867 (−0.010), Run 7 0.810 → 0.793 (−0.017). En ambos baja algo (2 de 2; por clase hasta −0.037, moneda del Run 7): los checkpoints originales son los que se usan. Hipótesis no probada de la leve bajada: la recalibración estima las estadísticas sin aumentos, mientras que el entrenamiento las estimó con ellos.
- **EXIF en el mismo marco correcto (Run 6 frente a Run 7, EXIF=1):** moneda **+0.170** (0.979 frente a 0.809), mAP +0.067. El efecto en moneda es grande frente al ruido de evaluación (cota por clase ≈ 0.02–0.03).
- **El 0.382 de moneda del Run 7 en su propio marco era sobre todo un artefacto de evaluación:** con las cajas alineadas sube a 0.809; bolígrafo y libro dan el mismo AP en ambos marcos (no tienen EXIF), laptop +0.001, mouse −0.016 (ruido).
- **Ambos modelos rinden mejor con imágenes orientadas** (Run 6: 0.877 frente a 0.671 sin orientar; Run 7: 0.810 frente a 0.728): la app debe entregar fotogramas orientados en cualquier caso.

**No demostrado:** el Run 6 también supera al Run 7 en clases sin EXIF (bolígrafo +0.035, libro +0.053, mouse +0.060, laptop +0.013). Con un run por condición no se separa de la variabilidad entre entrenamientos (no medida); un efecto indirecto de entrenar con etiquetas desalineadas es solo una hipótesis. Tampoco está aislado cuánto del salto se debe al BN correcto en sí y cuánto al entrenamiento más largo con LR decreciente que la `val_loss` fiable permite (hipótesis de §14.13).

**Limitaciones:** un solo run por condición; sin conjunto de prueba independiente (val eligió checkpoint y LR; resultados algo optimistas); bolígrafo sigue siendo la clase débil (0.50–0.55) en todas las condiciones.

**Estado de la decisión:** según el criterio acordado, **no se elige todavía el modelo para la app**. Datos para esa decisión: en el marco correcto el Run 6 es el mejor en todas las clases (0.877); requiere que el preprocesado de la app entregue imágenes orientadas, igual que lo necesitaría el Run 7 para rendir al máximo. Pendiente de decidir con el usuario.

### 14.15 Modelo elegido para la app: Run 6, exportado a TFLite y verificado (25-nov-2026)
**Decisión del usuario:** usar el **Run 6** (E6b-EXIF, `archivo/best_m6.keras`, SHA256 `24b06622…99d73`) en la app, sin cambiar código ni guías. Detalle completo en `diagnostico/export_m6/verificacion.txt`.

**Pasos (ejecutados por el usuario en WSL2, un comando a la vez):** (1) `cp archivo/best_m6.keras salida/best_model.keras` (el Run 7 que había en `salida/` ya estaba archivado); (2) evaluación de control `APLICAR_EXIF=1 python evaluate.py --score 0.5` → mAP 0.877, idéntica al Run 6; (3) `python export_tflite.py --cuantizar` → `salida/detector.tflite` (3.18 MB) y `salida/labels.txt` (avisos de memoria CUDA al inicio, sin efecto).

**Por qué en WSL:** el `.venv` de Windows (Keras 3.12.4) no carga con `load_model` los `.keras` guardados en WSL (Keras 3.16 nightly). Reconstruir la arquitectura con `construir_ssd()` + `load_weights` sí funciona (323/323 pesos; mismos AP por clase en Windows), pero el usuario prefirió exportar en WSL para no modificar `export_tflite.py`.

**Archivos entregables (copiados a `archivo/`, SHA256 idénticos a `salida/`):** `archivo/detector_m6.tflite` (SHA256 `634d326066df46a40ff8b400882845f778b912a408cb0af6cbc78f7a9e2f399f`, 3,180,240 bytes) y `archivo/labels_m6.txt` (`fondo, boligrafo, laptop, libro, moneda, mouse`; SHA256 `0d1cd819…edb85`).

**Verificación (Claude, `.venv` de Windows con el intérprete TFLite; `diagnostico/export_m6/verificar_tflite.py`):**
- Entrada `[1,256,256,3]` float32; salidas `[1,2046,6]` (scores) y `[1,2046,4]` (boxes), **en ese orden**: la app debe identificarlas por shape.
- Frente al modelo Keras en las 313 imágenes de val: |Δ cajas| medio 0.00017 (máx 0.0069), |Δ scores| medio 0.00001 (máx 0.033); misma clase top-1 en el 99.997 % de los anchors.
- **mAP@0.5 del `.tflite` = 0.877** (boli 0.552, laptop 0.995, libro 0.902, moneda 0.979, mouse 0.956; P 0.921, R 0.651), igual que el Keras: la cuantización dinámica no tiene coste medible.

**Requisitos para la app:** imagen **orientada** (sin orientar el mAP cae a 0.671, moneda 0.152); reescalado a 256×256 sin conservar proporción, RGB, [0,1] float32; por clase score ≥ 0.5 y NMS IoU 0.45; cajas (ymin, xmin, ymax, xmax) normalizadas. Pendiente de verificar en la app que los fotogramas llegan orientados.

**Limitaciones (heredadas de §14.13–14.14):** mAP medido en val (usado también para elegir checkpoint y LR), un solo run; bolígrafo es la clase débil (0.55); el rendimiento en condiciones reales de la app no se ha medido.

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
