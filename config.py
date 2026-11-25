"""Configuración central del proyecto.

Para adaptar el detector a otras 5 categorías basta con editar CLASES.
Todo lo demás (nº de clases, labels.txt, cabezas del modelo) se deriva de aquí.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- Clases
# El índice en esta lista == id de clase en los .txt de YOLO (0..4).
CLASES = ["boligrafo", "laptop", "libro", "moneda", "mouse"]
NUM_CLASES = len(CLASES)          # sin contar el fondo
NUM_CLASES_CON_FONDO = NUM_CLASES + 1   # índice 0 = fondo, 1..N = objetos

# ---------------------------------------------------------------- Rutas
RAIZ = Path(__file__).resolve().parent
DATA_DIR = RAIZ / "data"          # data/{train,val}/{images,labels}
SALIDA_DIR = RAIZ / "salida"      # modelos y exportaciones
MEJOR_MODELO = SALIDA_DIR / "best_model.keras"
TFLITE_PATH = SALIDA_DIR / "detector.tflite"
LABELS_PATH = SALIDA_DIR / "labels.txt"

# ---------------------------------------------------------------- Imagen
IMG_SIZE = 256                    # debe ser 256 (los mapas de características asumen esto)
MAX_CAJAS = 20                    # máx. objetos por imagen (se rellena con -1)
# Orientación EXIF (experimento E1). Las etiquetas están dibujadas sobre la imagen YA orientada,
# pero tf.io.decode_image devuelve los píxeles sin rotar. True = rotar/reflejar según la etiqueta
# EXIF antes del resize (en train, val y evaluación). False = comportamiento legacy (sin rotar).
# Se activa explícitamente con la variable de entorno APLICAR_EXIF=1 (p. ej. APLICAR_EXIF=1 python train.py).
APLICAR_EXIF = os.environ.get("APLICAR_EXIF", "0").strip() == "1"

# ---------------------------------------------------------------- BatchNorm (experimento E6b)
# MobileNetV2 de keras.applications usa momentum=0.999 en sus BatchNorm; con ~69 pasos por época las estadísticas
# móviles quedan desfasadas y el modo inferencia rinde mucho peor que el de entrenamiento (bitácora §14.9).
# None = valor de Keras (0.999, comportamiento legacy y de los Runs 1-5). Con la variable de entorno
# BN_MOMENTUM_BACKBONE=0.99 (p. ej. BN_MOMENTUM_BACKBONE=0.99 python train.py) model.py aplica ese valor SOLO a las
# BatchNorm del backbone; las de los bloques extra ya usan 0.99 (valor por defecto de Keras) y no se tocan.
# Revertir = no definir la variable. El valor queda guardado en el .keras, no hace falta repetirlo al evaluar.
_bn_mom = os.environ.get("BN_MOMENTUM_BACKBONE", "").strip()
BN_MOMENTUM_BACKBONE = float(_bn_mom) if _bn_mom else None
assert BN_MOMENTUM_BACKBONE is None or 0.0 < BN_MOMENTUM_BACKBONE < 1.0, "BN_MOMENTUM_BACKBONE debe estar en (0, 1)"

# ---------------------------------------------------------------- Anchors SSD
# Tamaño de cada mapa de características (stride 16, 32, 64, 128, 256).
FEATURE_MAPS = [IMG_SIZE // 16, IMG_SIZE // 32, IMG_SIZE // 64,
                IMG_SIZE // 128, IMG_SIZE // 256]          # 16, 8, 4, 2, 1
ESCALA_MIN, ESCALA_MAX = 0.07, 0.95                         # 0.07 para capturar objetos pequeños (monedas)
ANCHORS_POR_CELDA = 6             # ratios 1, 1' (escala extra), 2, 1/2, 3, 1/3 (bolígrafos alargados)
VARIANZAS = (0.1, 0.2)            # (centro, tamaño) para codificar offsets
UMBRAL_IOU_POSITIVO = 0.5

# ---------------------------------------------------------------- Entrenamiento
BATCH_SIZE = 16
EPOCAS = 150
LR_INICIAL = 1e-3
RATIO_NEGATIVOS = 3               # hard negative mining 3:1
PACIENCIA = 30                    # épocas sin mejora antes de detenerse (EarlyStopping)

# ---------------------------------------------------------------- Inferencia / evaluación
UMBRAL_SCORE = 0.5
UMBRAL_NMS = 0.45
