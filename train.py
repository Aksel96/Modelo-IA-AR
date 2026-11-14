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
    args = ap.parse_args()
    C.BATCH_SIZE = args.batch
    C.SALIDA_DIR.mkdir(parents=True, exist_ok=True)

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
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                         restore_best_weights=True),
    ]
    modelo.fit(ds_train, validation_data=ds_val, epochs=args.epocas, callbacks=callbacks)
    print(f"Mejor modelo guardado en {C.MEJOR_MODELO}")


if __name__ == "__main__":
    main()
