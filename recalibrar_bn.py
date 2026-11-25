"""Recalibración de las estadísticas móviles de BatchNorm de un checkpoint (E6a).  NO entrena.

Motivo (bitácora §14.9): los BN del backbone usan momentum=0.999 y con ~69 pasos por época las medias/varianzas móviles
quedan desfasadas respecto a los pesos, así que el modelo en inferencia rinde mucho peor que con BN de lote. Esta
herramienta hace pasadas hacia adelante (training=True, sin gradientes ni optimizador) sobre las imágenes de TRAIN para
reestimar solo `moving_mean` y `moving_variance`; todo lo demás (pesos entrenables, gamma/beta) queda idéntico bit a bit.

Garantías (las comprueba el propio script antes y después de guardar, y tests/test_recalibrar_bn.py):
  * usa exclusivamente imágenes de data/train, sin aumentos, con la misma carga que el entrenamiento (incluida la
    orientación EXIF si APLICAR_EXIF=1);
  * ningún peso distinto de las estadísticas móviles cambia;
  * el checkpoint original no se toca (se comprueba su SHA256 antes y después) y el derivado nunca sobrescribe un archivo
    existente;
  * es reproducible: mismo checkpoint + mismos parámetros + misma semilla = mismas estadísticas (en CPU; en GPU puede
    haber diferencias de redondeo);
  * el momentum de cada capa BN se restaura al valor original antes de guardar (el derivado conserva su configuración).

La bandera APLICAR_EXIF debe ser la misma con la que se entrenó el checkpoint (las estadísticas dependen de los píxeles);
se registra en el archivo auxiliar `<derivado>.recal.json`.

Uso (desde WSL2, con la bandera con la que se entrenó):
    APLICAR_EXIF=1 python recalibrar_bn.py --modelo archivo/best_m5.keras --salida archivo/best_m5_bnrecal.keras
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

import config as C
import data as D

MOMENTUM_RECAL = 0.9     # el que se validó en d6; con 69 pasos x 3 pasadas el valor inicial pesa 0.9**207 ~ 3e-10


def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def capas_bn(modelo):
    """Todas las capas BatchNormalization (también dentro de submodelos), en orden estable."""
    res = []
    for capa in modelo.layers:
        if isinstance(capa, tf.keras.layers.BatchNormalization):
            res.append(capa)
        elif hasattr(capa, "layers"):
            res += capas_bn(capa)
    return res


def instantanea(modelo):
    """Lista [(nombre, es_estadistica_bn, ndarray)] de TODOS los pesos del modelo, en el orden de `modelo.weights`."""
    ids_est = set()
    for capa in capas_bn(modelo):
        ids_est.add(id(capa.moving_mean))
        ids_est.add(id(capa.moving_variance))
    salida = []
    for i, w in enumerate(modelo.weights):
        nombre = f"{i}:{getattr(w, 'path', w.name)}"
        salida.append((nombre, id(w) in ids_est, np.array(w.numpy(), copy=True)))
    return salida


def comparar(antes, despues):
    """Compara dos instantáneas. Devuelve dict con los índices cambiados de cada tipo."""
    if len(antes) != len(despues):
        raise ValueError(f"número de pesos distinto: {len(antes)} vs {len(despues)}")
    mal, est_cambiadas, est_iguales = [], [], []
    for (na, ea, a), (nb, eb, b) in zip(antes, despues):
        if na != nb or ea != eb or a.shape != b.shape:
            raise ValueError(f"estructura distinta en {na} / {nb}")
        igual = np.array_equal(a, b)
        if not ea and not igual:
            mal.append(na)                       # un peso que NO es estadística de BN cambió: prohibido
        elif ea:
            (est_iguales if igual else est_cambiadas).append(na)
    return {"pesos_no_estadistica_cambiados": mal, "estadisticas_cambiadas": est_cambiadas,
            "estadisticas_sin_cambio": est_iguales}


def rutas_train():
    """Rutas de las imágenes de data/train (únicas usadas para recalibrar)."""
    rutas, _ = D.listar_muestras("train")
    return rutas


def recalibrar(modelo, rutas, pasadas=3, momentum=MOMENTUM_RECAL, semilla=0, max_lotes=None):
    """Actualiza EN SITIO las estadísticas móviles de BN de `modelo` con pasadas hacia adelante sobre `rutas`.

    Cada pasada recorre las imágenes en un orden aleatorio fijado por `semilla + pasada` (reproducible), sin aumentos.
    Restaura el momentum original de cada capa al terminar. Devuelve el nº de pasos de lote realizados.
    """
    bns = capas_bn(modelo)
    if not bns:
        raise ValueError("el modelo no tiene capas BatchNormalization")
    momentum_original = [capa.momentum for capa in bns]
    pasos = 0
    try:
        for capa in bns:
            capa.momentum = momentum
        for p in range(pasadas):
            orden = np.random.RandomState(semilla + p).permutation(len(rutas))
            ds = D.dataset_solo_imagenes([rutas[i] for i in orden])
            for k, x in enumerate(ds):
                if max_lotes is not None and k >= max_lotes:
                    break
                modelo(x, training=True)           # solo actualiza moving_mean/moving_variance; sin gradientes
                pasos += 1
    finally:
        for capa, m in zip(bns, momentum_original):
            capa.momentum = m
    return pasos


def ejecutar(modelo_path, salida_path, pasadas=3, momentum=MOMENTUM_RECAL, semilla=0, max_lotes=None, rutas=None,
             log=print):
    """Flujo completo: carga, recalibra, verifica, guarda el derivado y comprueba el original. Devuelve el informe."""
    modelo_path, salida_path = Path(modelo_path), Path(salida_path)
    if not modelo_path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {modelo_path}")
    if salida_path.resolve() == modelo_path.resolve():
        raise ValueError("--salida no puede ser el mismo archivo que --modelo (el original no se sobrescribe)")
    if salida_path.exists():
        raise FileExistsError(f"{salida_path} ya existe; no se sobrescribe (elija otro nombre o bórrelo a mano)")
    if rutas is None:
        rutas = rutas_train()

    sha_antes = sha256_archivo(modelo_path)
    modelo = tf.keras.models.load_model(str(modelo_path), compile=False)
    antes = instantanea(modelo)
    pasos = recalibrar(modelo, rutas, pasadas, momentum, semilla, max_lotes)
    despues = instantanea(modelo)

    cmp_ = comparar(antes, despues)
    if cmp_["pesos_no_estadistica_cambiados"]:
        raise RuntimeError("Un peso distinto de las estadísticas de BN cambió; no se guarda nada: "
                           + ", ".join(cmp_["pesos_no_estadistica_cambiados"][:5]))
    if not cmp_["estadisticas_cambiadas"]:
        raise RuntimeError("Ninguna estadística de BN cambió; no se guarda nada")

    salida_path.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(str(salida_path))
    # El derivado guardado debe reproducir exactamente los pesos en memoria.
    releido = instantanea(tf.keras.models.load_model(str(salida_path), compile=False))
    if comparar(despues, releido)["pesos_no_estadistica_cambiados"] or comparar(despues, releido)["estadisticas_cambiadas"]:
        raise RuntimeError("El derivado guardado no coincide con el modelo en memoria")
    sha_despues = sha256_archivo(modelo_path)
    if sha_despues != sha_antes:
        raise RuntimeError("¡El checkpoint original cambió durante la ejecución!")

    n_est = sum(1 for _, e, _ in antes if e)
    informe = {
        "origen": str(modelo_path), "origen_sha256": sha_antes, "derivado": str(salida_path),
        "derivado_sha256": sha256_archivo(salida_path), "aplicar_exif": bool(C.APLICAR_EXIF),
        "pasadas": pasadas, "momentum_recalibracion": momentum, "semilla": semilla, "max_lotes": max_lotes,
        "imagenes_train_disponibles": len(rutas), "pasos_de_lote": pasos, "batch_size": C.BATCH_SIZE,
        "pesos_totales": len(antes), "estadisticas_bn": n_est,
        "estadisticas_cambiadas": len(cmp_["estadisticas_cambiadas"]),
        "estadisticas_sin_cambio": len(cmp_["estadisticas_sin_cambio"]),
        "pesos_no_estadistica_cambiados": 0,
        "tensorflow": tf.__version__, "keras": tf.keras.__version__, "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    salida_path.with_suffix(".recal.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Origen   : {modelo_path}  SHA256 {sha_antes} (sin cambios tras la ejecución)")
    log(f"Derivado : {salida_path}  SHA256 {informe['derivado_sha256']}")
    log(f"APLICAR_EXIF={int(C.APLICAR_EXIF)} (debe ser el de entrenamiento) | train: {len(rutas)} imágenes | "
        f"{pasadas} pasadas, {pasos} pasos de lote (batch {C.BATCH_SIZE}) | momentum de recalibración {momentum} | semilla {semilla}")
    log(f"Pesos: {len(antes)} en total; {n_est} son estadísticas de BN ({informe['estadisticas_cambiadas']} cambiadas, "
        f"{informe['estadisticas_sin_cambio']} sin cambio); pesos que NO son estadísticas y cambiaron: 0")
    log(f"Auxiliar : {salida_path.with_suffix('.recal.json')}")
    return informe


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modelo", required=True, help="checkpoint .keras de origen (no se modifica)")
    ap.add_argument("--salida", required=True, help="checkpoint derivado a crear (no debe existir)")
    ap.add_argument("--pasadas", type=int, default=3)
    ap.add_argument("--momentum", type=float, default=MOMENTUM_RECAL)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--max-lotes", type=int, default=None, help="solo para pruebas: lotes máximos por pasada")
    a = ap.parse_args()
    try:
        ejecutar(a.modelo, a.salida, a.pasadas, a.momentum, a.semilla, a.max_lotes)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        sys.exit(f"Error: {e}")
