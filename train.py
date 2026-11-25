"""Entrenamiento desde cero.  Uso: python train.py [--epocas N] [--batch N]"""
import argparse

import tensorflow as tf

import config as C
from data import construir_dataset
from losses import perdida_ssd
from model import construir_ssd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=C.EPOCAS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR_INICIAL)
    ap.add_argument("--paciencia", type=int, default=C.PACIENCIA,
                    help="Épocas sin mejora en val_loss antes de detener el entrenamiento")
    ap.add_argument("--semilla", type=int, default=None,
                    help="Fija las semillas de Python/NumPy/TensorFlow (pesos iniciales y orden de los datos) para que "
                         "dos entrenamientos comparables partan igual. Sin esta opción, no se fija ninguna (como antes).")
    args = ap.parse_args()
    C.BATCH_SIZE = args.batch
    C.SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    if args.semilla is not None:
        tf.keras.utils.set_random_seed(args.semilla)
    print(f"Configuración: APLICAR_EXIF={int(C.APLICAR_EXIF)} | BN_MOMENTUM_BACKBONE={C.BN_MOMENTUM_BACKBONE} "
          f"(None = 0.999 de Keras) | semilla={args.semilla} | batch={args.batch} | lr={args.lr}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs detectadas: {gpus}" if gpus else "AVISO: TensorFlow NO ve la GPU (entrenará en CPU)")

    ds_train = construir_dataset("train", entrenamiento=True)
    ds_val = construir_dataset("val", entrenamiento=False)

    modelo = construir_ssd()
    modelo.summary(line_length=100)
    modelo.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=perdida_ssd)

    callbacks = [
        # Guarda solo el mejor modelo según la pérdida de validación.
        tf.keras.callbacks.ModelCheckpoint(str(C.MEJOR_MODELO), monitor="val_loss",
                                           save_best_only=True, verbose=1),
        # Reduce el learning rate cuando la validación se estanca.
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.paciencia,
                                         restore_best_weights=True),
        # Historial por época (loss, val_loss, lr) para diagnosticar el entrenamiento.
        tf.keras.callbacks.CSVLogger(str(C.SALIDA_DIR / "historial.csv")),
    ]
    modelo.fit(ds_train, validation_data=ds_val, epochs=args.epocas, callbacks=callbacks)
    print(f"Mejor modelo guardado en {C.MEJOR_MODELO}")


if __name__ == "__main__":
    main()
