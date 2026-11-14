"""Configuración central del proyecto.

Para adaptar el detector a otras 5 categorías basta con editar CLASES.
Todo lo demás (nº de clases, labels.txt, cabezas del modelo) se deriva de aquí.
"""
from pathlib import Path

# ---------------------------------------------------------------- Clases
# El índice en esta lista == id de clase en los .txt de YOLO (0..4).
CLASES = ["moneda", "laptop", "boligrafo", "raton", "libro"]
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

# ---------------------------------------------------------------- Anchors SSD
# Tamaño de cada mapa de características (stride 16, 32, 64, 128, 256).
FEATURE_MAPS = [IMG_SIZE // 16, IMG_SIZE // 32, IMG_SIZE // 64,
                IMG_SIZE // 128, IMG_SIZE // 256]          # 16, 8, 4, 2, 1
ESCALA_MIN, ESCALA_MAX = 0.10, 0.95
ANCHORS_POR_CELDA = 4             # ratios 1, 1' (escala extra), 2, 1/2
VARIANZAS = (0.1, 0.2)            # (centro, tamaño) para codificar offsets
UMBRAL_IOU_POSITIVO = 0.5

# ---------------------------------------------------------------- Entrenamiento
BATCH_SIZE = 16
EPOCAS = 150
LR_INICIAL = 1e-3
RATIO_NEGATIVOS = 3               # hard negative mining 3:1

# ---------------------------------------------------------------- Inferencia / evaluación
UMBRAL_SCORE = 0.5
UMBRAL_NMS = 0.45
