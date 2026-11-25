"""Tests del cambio de momentum de BatchNorm del backbone (E6b) y de la opción --semilla.  Uso, desde la raíz del repo:

    python -B -m unittest tests.test_bn_momentum -v

No entrenan (nunca se importa ni se ejecuta train.py: solo se inspecciona su código con `ast`) ni tocan archivos del
proyecto; los modelos son de pesos aleatorios.
"""
import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

import config as C
from model import construir_ssd

RAIZ = Path(__file__).resolve().parents[1]


def _construir(momentum):
    prev = C.BN_MOMENTUM_BACKBONE
    C.BN_MOMENTUM_BACKBONE = momentum
    try:
        return construir_ssd()
    finally:
        C.BN_MOMENTUM_BACKBONE = prev


def _bn(capa_o_modelo):
    res = []
    for c in capa_o_modelo.layers:
        if isinstance(c, tf.keras.layers.BatchNormalization):
            res.append(c)
        elif hasattr(c, "layers"):
            res += _bn(c)
    return res


def _separar(modelo):
    """(BN del backbone, BN de los bloques extra)."""
    back = _bn(modelo.get_layer("mobilenetv2_scratch"))
    extra = [c for c in modelo.layers if isinstance(c, tf.keras.layers.BatchNormalization)]
    return back, extra


class TestMomentumBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = _construir(None)
        cls.nuevo = _construir(0.99)

    def test_por_defecto_no_cambia_nada(self):
        back, extra = _separar(self.legacy)
        self.assertEqual({c.momentum for c in back}, {0.999})
        self.assertEqual({c.momentum for c in extra}, {0.99})

    def test_con_099_solo_cambia_el_backbone(self):
        back, extra = _separar(self.nuevo)
        self.assertEqual({c.momentum for c in back}, {0.99})
        self.assertEqual({c.momentum for c in extra}, {0.99})           # ya eran 0.99: sin cambio
        bl, el = _separar(self.legacy)
        self.assertEqual((len(back), len(extra)), (len(bl), len(el)))
        self.assertEqual((len(back), len(extra)), (52, 6))               # 58 BN en total
        self.assertEqual(len(_bn(self.nuevo)), 58)

    def test_ningun_otro_atributo_de_bn_cambia(self):
        for a, b in zip(_bn(self.legacy), _bn(self.nuevo)):
            self.assertEqual((a.name, a.epsilon, a.axis, a.center, a.scale), (b.name, b.epsilon, b.axis, b.center, b.scale))

    def test_misma_arquitectura_y_misma_inferencia(self):
        """El momentum no cambia pesos, formas ni la salida en inferencia."""
        self.assertEqual([w.shape for w in self.legacy.weights], [w.shape for w in self.nuevo.weights])
        self.assertEqual(len(self.legacy.trainable_weights), len(self.nuevo.trainable_weights))
        self.nuevo.set_weights(self.legacy.get_weights())
        x = np.random.RandomState(0).uniform(0, 1, (2, C.IMG_SIZE, C.IMG_SIZE, 3)).astype("float32")
        ya, yb = self.legacy(x, training=False).numpy(), self.nuevo(x, training=False).numpy()
        self.assertTrue(np.array_equal(ya, yb))

    def test_el_momentum_actua_de_verdad_en_la_actualizacion(self):
        """Tras un paso con training=True desde las mismas estadísticas iniciales: (1-m) multiplica la media del lote."""
        a, b = _construir(None), _construir(0.99)
        b.set_weights(a.get_weights())
        x = np.random.RandomState(1).uniform(0, 1, (4, C.IMG_SIZE, C.IMG_SIZE, 3)).astype("float32")
        a(x, training=True), b(x, training=True)
        ca, cb = _separar(a)[0][0], _separar(b)[0][0]                      # primera BN del backbone (misma entrada)
        # estadísticas iniciales: media 0, varianza 1  ->  nueva = m*inicial + (1-m)*lote
        self.assertTrue(np.allclose(cb.moving_mean.numpy(), 10 * ca.moving_mean.numpy(), rtol=1e-3, atol=1e-8))
        self.assertTrue(np.allclose(cb.moving_variance.numpy() - 1, 10 * (ca.moving_variance.numpy() - 1), rtol=1e-3, atol=1e-6))
        self.assertGreater(float(np.max(np.abs(cb.moving_mean.numpy()))), 0)

    def test_el_valor_se_guarda_y_se_recarga_en_el_keras(self):
        with tempfile.TemporaryDirectory() as t:
            ruta = str(Path(t) / "m.keras")
            self.nuevo.save(ruta)
            cargado = tf.keras.models.load_model(ruta, compile=False)
        back, extra = _separar(cargado)
        self.assertEqual({c.momentum for c in back}, {0.99})
        self.assertEqual({c.momentum for c in extra}, {0.99})


class TestConfigEntorno(unittest.TestCase):
    @staticmethod
    def _leer(valor):
        env = {k: v for k, v in os.environ.items() if k != "BN_MOMENTUM_BACKBONE"}
        if valor is not None:
            env["BN_MOMENTUM_BACKBONE"] = valor
        return subprocess.run([sys.executable, "-B", "-c", "import config as C; print(repr(C.BN_MOMENTUM_BACKBONE))"],
                              cwd=RAIZ, env=env, capture_output=True, text=True)

    def test_sin_variable_o_vacia_es_none(self):
        for v in (None, "", "  "):
            r = self._leer(v)
            self.assertEqual((r.returncode, r.stdout.strip()), (0, "None"), r.stderr[-300:])

    def test_variable_valida(self):
        r = self._leer("0.99")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "0.99"))

    def test_variable_fuera_de_rango_es_rechazada(self):
        for v in ("1.5", "0", "-0.1", "1"):
            self.assertNotEqual(self._leer(v).returncode, 0, v)


class TestSemilla(unittest.TestCase):
    def test_misma_semilla_mismos_pesos_iniciales(self):
        tf.keras.utils.set_random_seed(123)
        a = construir_ssd().get_weights()
        tf.keras.utils.set_random_seed(123)
        b = construir_ssd().get_weights()
        tf.keras.utils.set_random_seed(124)
        c = construir_ssd().get_weights()
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(all(np.array_equal(x, y) for x, y in zip(a, c)))

    def test_el_momentum_no_altera_los_pesos_iniciales_con_la_misma_semilla(self):
        """Condición para comparar E6b-EXIF y E6b-Legacy: con la misma semilla parten de los mismos pesos."""
        tf.keras.utils.set_random_seed(7)
        a = _construir(None).get_weights()
        tf.keras.utils.set_random_seed(7)
        b = _construir(0.99).get_weights()
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))

    def test_train_py_ofrece_semilla_opcional_sin_valor_por_defecto(self):
        """Inspección estática (train.py NO se importa ni se ejecuta)."""
        arbol = ast.parse((RAIZ / "train.py").read_text(encoding="utf-8"))
        llamadas = [n for n in ast.walk(arbol) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                    and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--semilla"]
        self.assertEqual(len(llamadas), 1)
        kw = {k.arg: k.value for k in llamadas[0].keywords}
        self.assertIsInstance(kw["default"], ast.Constant)
        self.assertIsNone(kw["default"].value)                        # sin la opción, no se fija ninguna semilla
        fuente = (RAIZ / "train.py").read_text(encoding="utf-8")
        self.assertIn("set_random_seed(args.semilla)", fuente)


if __name__ == "__main__":
    unittest.main()
