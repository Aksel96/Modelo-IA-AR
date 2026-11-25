# Guía de reentrenamiento en WSL2 con GPU (RTX)

Continúa desde el **paso 5 (Entrenar)** de [GUIA.md](GUIA.md). Estado de partida:

- WSL2 ya detecta la GPU (NVIDIA RTX serie 30/40/50).
- Proyecto en `/mnt/c/...` (disco de Windows): hay que moverlo, ver paso 1.
- Dataset: 100–300 imágenes por clase (por debajo de lo recomendado, ≥ 300).
- Primera ejecución (Windows, CPU): mejor modelo en la época 22 (`val_loss` = 9.2566). Se detuvo en la 42 con `val_loss` = 10.5427 y `learning_rate` = 1.25e-4. No se midió mAP.
- Decisión: **reiniciar desde cero** (`weights=None`) y conservar el modelo anterior para comparar.

## 0. Qué pasó en la primera ejecución

- **Las 42 épocas son normales.** 22 + 20 = 42. `EarlyStopping(patience=20)` paró el entrenamiento al pasar 20 épocas sin mejorar `val_loss`. No fue un fallo.
- **`val_loss` subió de 9.26 a 10.54 mientras la pérdida de entrenamiento seguramente seguía bajando.** Eso es sobreajuste: el modelo memoriza el train y deja de generalizar.
- **El LR se redujo tres veces** (1e-3 → 5e-4 → 2.5e-4 → 1.25e-4). `ReduceLROnPlateau` lo bajó sin que la validación mejorara.
- **La GPU acelera cada época, pero no corrige el sobreajuste.** Con la misma cantidad de datos puede parar en una época parecida. Lo que sí cambia es que probar ideas pasa de 1–5 min por época a pocos segundos.
- **Subir la paciencia a 40 solo ayuda a veces.** `val_loss` desde cero es muy ruidoso al inicio, y a veces mejora tras una meseta larga. Si la curva sigue subiendo, parar pronto era lo correcto. Más datos es lo que más mueve el resultado.

## 1. Mover el proyecto al disco de Linux de WSL

Leer miles de imágenes desde `/mnt/c` es mucho más lento, y con GPU suele ser el cuello de botella. Cópialo al sistema de archivos de WSL (ext4):

```bash
cd ~
cp -r /mnt/c/RUTA/A/Modelo-IA-AR ~/Modelo-IA-AR
cd ~/Modelo-IA-AR
rm -rf .venv   # el venv de Windows no sirve en Linux
```

Cambia `RUTA/A/Modelo-IA-AR` por tu ruta real. Si el proyecto está en Git, también puedes clonarlo directamente en `~`.

## 2. Entorno Python con TensorFlow + CUDA

Usa Python 3.10–3.12:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "tensorflow[and-cuda]" numpy pillow
```

`tensorflow[and-cuda]` trae las librerías CUDA/cuDNN dentro del venv. Solo necesitas el driver NVIDIA de Windows (ya lo tienes si WSL ve la GPU). No instales el CUDA Toolkit dentro de WSL.

Comprueba:

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Debe salir una lista con `PhysicalDevice(name='/physical_device:GPU:0'...)`. Si sale `[]`:

| Síntoma | Qué revisar |
|---|---|
| `nvidia-smi` falla en WSL | Actualiza el driver NVIDIA en **Windows** y ejecuta `wsl --shutdown` en PowerShell. |
| `nvidia-smi` funciona pero TF no ve GPU | Instalaste `tensorflow` sin `[and-cuda]`, o el venv no está activo. |
| Warnings de `libdevice`, `ptxas` o XLA | Suelen ser inofensivos. Revisa que el entrenamiento arranque. |

## 3. Archivar el entrenamiento anterior

`train.py` sobrescribe `salida/best_model.keras`. Guarda antes el modelo de la primera ejecución (época 22):

```bash
mkdir -p archivo
[ -d salida ] && mv salida archivo/salida_run1_cpu
```

Si traes el `best_model.keras` anterior desde Windows, cópialo a `archivo/salida_run1_cpu/`. Sirve de comparación en el paso 8.

## 4. Verificar el dataset

```bash
python verificar_dataset.py
```

Revisa:

- Que diga `Dataset OK` y no haya `[ERROR]`.
- **Balance**: ninguna clase debería tener menos de la mitad de cajas que la mayor. Si una clase tiene muy pocas, la métrica por clase será mala.
- **Val**: ≥ 40–50 cajas por clase para que `val_loss` no sea puro ruido.
- **Sin fugas**: no deben repetirse fotos casi idénticas entre `train` y `val`. Si ocurre, `val_loss` se verá mejor de lo real.

Con 100–300 imágenes por clase, **añadir 100–200 imágenes por clase es la mejora que más sube el mAP**. Mejora la precisión más que cualquier ajuste de hiperparámetros.

## 5. Prueba corta: confirmar que la GPU se usa

Primero ejecuta unas pocas épocas y mide el tiempo. En otra terminal de WSL, abre `nvidia-smi`:

```bash
watch -n 1 nvidia-smi
```

```bash
python train.py --epocas 3 --batch 32
```

Debe imprimir `GPUs detectadas: [PhysicalDevice...]`.

- **Uso de GPU (`nvidia-smi`) con picos de 60–100 %**: bien.
- **Casi siempre < 30 %**: el cuello de botella son los datos. Comprueba que el proyecto esté en `~` y no en `/mnt/c`.
- **`Out of memory`**: baja a `--batch 16`.

Anota el tiempo por época. Debería ser claramente menor que los 1–5 min de la CPU.

## 6. Entrenamiento real

Con 8 GB o más de VRAM, `--batch 32` va bien. Si tienes menos, usa `--batch 16`.

```bash
nohup python train.py --epocas 200 --batch 32 --paciencia 40 > salida_entrenamiento.log 2>&1 &
tail -f salida_entrenamiento.log
```

- `nohup ... &` mantiene el entrenamiento aunque cierres la terminal. Con `tail -f` sigues el progreso; `Ctrl+C` solo cierra el `tail`.
- `--paciencia 40` (el argumento real de `train.py`) es el cambio que pediste: ahora se detiene tras 40 épocas sin mejora (antes 20).
- `--epocas 200` sube el tope para que haya margen.
- El mejor modelo se guarda en `salida/best_model.keras`, y el historial por época en `salida/historial.csv`.
- **Orientación EXIF:** para entrenar con la corrección, antepón `APLICAR_EXIF=1` (p. ej. `APLICAR_EXIF=1 nohup python train.py ...`) y úsala también al evaluar. Ver README.

Para parar un entrenamiento (si hay que cortarlo): `pkill -f train.py`. El mejor modelo hasta ese momento ya está guardado.

Mientras corre, no cierres WSL ni apagues el PC. Si Windows suspende el equipo, el entrenamiento se interrumpe.

## 7. Leer las curvas

```bash
python - <<'EOF'
import csv
f = list(csv.DictReader(open("salida/historial.csv")))
mejor = min(f, key=lambda r: float(r["val_loss"]))
print("épocas:", len(f), "| mejor época:", int(mejor["epoch"]) + 1, "| val_loss:", mejor["val_loss"])
print("loss (train) final:", f[-1]["loss"], "| val_loss final:", f[-1]["val_loss"])
EOF
```

| Lo que ves | Diagnóstico | Qué hacer |
|---|---|---|
| `loss` y `val_loss` bajan juntas | Va bien | Dejar que termine. |
| `loss` baja, `val_loss` sube después de pocas épocas | Sobreajuste (como en la primera ejecución) | Más datos. Si no hay, más aumento de datos. |
| Ambas altas y planas | Poco aprendizaje | Revisa etiquetas, `verificar_dataset.py`, y prueba `--lr 5e-4`. |
| `val_loss` en picos muy irregulares | Val pequeño o muy ruidoso | Más imágenes de validación. |

Sobre la primera ejecución: `val_loss` en 9.26 es alto. Un entrenamiento sano suele terminar mucho más abajo. Eso apunta a datos insuficientes o a etiquetas por revisar, más que a la CPU o la GPU.

## 8. Evaluar y comparar con el modelo anterior

La primera ejecución no tiene mAP. Mídelo ahora en las dos para compararlas con el mismo criterio:

```bash
python evaluate.py --score 0.5            # modelo nuevo (salida/best_model.keras)
# Si el modelo se entrenó con EXIF:  APLICAR_EXIF=1 python evaluate.py --score 0.5
```

Para el modelo anterior, copia temporalmente su `best_model.keras` a `salida/`, evalúa, y restaura el nuevo. O renómbralos: `best_nuevo.keras` y `best_viejo.keras`.

Compara mAP@0.5, Precision y Recall por clase. Un `val_loss` más bajo suele implicar mejor mAP, pero no siempre.

## 9. Exportar a TFLite

Solo si el mAP es aceptable para tu app de RA:

```bash
python export_tflite.py --cuantizar
```

Genera `salida/detector.tflite` y `salida/labels.txt`.

## 10. Copiar resultados a Windows

```bash
mkdir -p /mnt/c/RUTA/DESTINO
cp salida/best_model.keras salida/detector.tflite salida/labels.txt salida/historial.csv /mnt/c/RUTA/DESTINO/
```

Lo nuevo se queda en `~/Modelo-IA-AR`. Si ese disco virtual de WSL se borra, se pierde. Copia el dataset y los modelos a Windows o Git cada cierto tiempo.

## Si el resultado sigue siendo pobre

En este orden de impacto:

1. **Más datos y más variedad** (fondos, luz, distancias, objetos parciales). Es lo que más ayuda al entrenar desde cero.
2. **Revisar etiquetas** con [GUIA_ETIQUETADO.md](GUIA_ETIQUETADO.md): cajas ajustadas y clases correctas.
3. **Más aumento de datos** en `data.py` (hoy solo flip, brillo y contraste): recortes y escala aleatoria.
4. Probar `--lr 5e-4` si el entrenamiento es inestable al principio.
5. Mixed precision (float16) para acelerar más. No tiene sentido hasta que el cuello de botella sean los datos y no el cómputo.
