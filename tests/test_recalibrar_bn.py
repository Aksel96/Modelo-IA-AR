"""Tests de la herramienta de recalibración de BatchNorm (E6a).  Uso, desde la raíz del repo:

    python -B -m unittest tests.test_recalibrar_bn -v

No entrenan ni tocan archivos del proyecto: usan un modelo SSD con pesos aleatorios guardado en un directorio temporal y
muy pocos lotes de train (el objetivo es verificar las garantías de la herramienta, no la calidad del modelo).
"""
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
import data as D
import evaluate as E
import recalibrar_bn as R
from model import construir_modelo_inferencia, construir_ssd

RAIZ = Path(__file__).resolve().parents[1]
LOTES = 2          # lotes por pasada en los tests (32 imágenes)


def _hay_train():
    d = C.DATA_DIR / "train" / "images"
    return d.exists() and any(d.glob("*.jp*g"))


def _modelo_con_estadisticas_no_triviales():
    """Modelo aleatorio con BN a medias: para que 'recalibrar' tenga algo que cambiar y no empiece en media 0 / var 1."""
    m = construir_ssd()
    rng = np.random.RandomState(123)
    for capa in R.capas_bn(m):
        capa.moving_mean.assign(rng.normal(0, 0.5, capa.moving_mean.shape).astype("float32"))
        capa.moving_variance.assign(rng.uniform(0.5, 2.0, capa.moving_variance.shape).astype("float32"))
    return m


@unittest.skipUnless(_hay_train(), "dataset de train no disponible")
class TestRecalibrarBN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.orig = cls.dir / "original.keras"
        _modelo_con_estadisticas_no_triviales().save(str(cls.orig))
        cls.sha_orig = R.sha256_archivo(cls.orig)
        cls.rutas = R.rutas_train()
        cls.x = next(iter(D.dataset_solo_imagenes(cls.rutas[:8])))      # un lote fijo para comparar salidas
        cls.deriv = cls.dir / "derivado.keras"
        cls.informe = R.ejecutar(cls.orig, cls.deriv, pasadas=1, max_lotes=LOTES, log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _cargar(ruta):
        return tf.keras.models.load_model(str(ruta), compile=False)

    # ---- 1. integridad de los pesos entrenables
    def test_pesos_entrenables_identicos_bit_a_bit(self):
        a, b = self._cargar(self.orig), self._cargar(self.deriv)
        ids_a = {id(v) for capa in R.capas_bn(a) for v in (capa.moving_mean, capa.moving_variance)}
        n = 0
        for va, vb in zip(a.weights, b.weights):
            if id(va) in ids_a:
                continue
            n += 1
            self.assertTrue(np.array_equal(va.numpy(), vb.numpy()), f"cambió {va.path}")
        self.assertGreater(n, 100)                      # hay muchos pesos no-estadística (convoluciones, gamma, beta...)
        self.assertEqual(len(a.weights), len(b.weights))
        # gamma y beta de BN son entrenables y deben seguir idénticos
        for ca, cb in zip(R.capas_bn(a), R.capas_bn(b)):
            self.assertTrue(np.array_equal(ca.gamma.numpy(), cb.gamma.numpy()))
            self.assertTrue(np.array_equal(ca.beta.numpy(), cb.beta.numpy()))

    # ---- 2. solo cambian las estadísticas móviles
    def test_solo_cambian_las_estadisticas_moviles(self):
        a, b = self._cargar(self.orig), self._cargar(self.deriv)
        for ca, cb in zip(R.capas_bn(a), R.capas_bn(b)):
            cambio = (not np.array_equal(ca.moving_mean.numpy(), cb.moving_mean.numpy())
                      or not np.array_equal(ca.moving_variance.numpy(), cb.moving_variance.numpy()))
            self.assertTrue(cambio, f"la capa {ca.name} no cambió")
        cmp_ = R.comparar(R.instantanea(a), R.instantanea(b))
        self.assertEqual(cmp_["pesos_no_estadistica_cambiados"], [])
        self.assertEqual(len(cmp_["estadisticas_cambiadas"]), 2 * len(R.capas_bn(a)))

    def test_momentum_original_se_conserva_en_el_derivado(self):
        for ca, cb in zip(R.capas_bn(self._cargar(self.orig)), R.capas_bn(self._cargar(self.deriv))):
            self.assertEqual(ca.momentum, cb.momentum)

    def test_comparar_detecta_un_peso_modificado(self):
        a = self._cargar(self.orig)
        antes = R.instantanea(a)
        conv = next(w for w in a.weights if len(w.shape) == 4)          # un kernel de convolución
        conv.assign(conv.numpy() + 1e-3)
        mal = R.comparar(antes, R.instantanea(a))["pesos_no_estadistica_cambiados"]
        self.assertEqual(len(mal), 1)

    # ---- 3. integridad del original y política de no sobrescribir
    def test_original_intacto(self):
        self.assertEqual(R.sha256_archivo(self.orig), self.sha_orig)
        self.assertEqual(self.informe["origen_sha256"], self.sha_orig)
        self.assertNotEqual(self.informe["derivado_sha256"], self.sha_orig)

    def test_no_sobrescribe_ni_acepta_el_mismo_archivo(self):
        with self.assertRaises(ValueError):
            R.ejecutar(self.orig, self.orig, max_lotes=1, log=lambda *_: None)
        with self.assertRaises(FileExistsError):
            R.ejecutar(self.orig, self.deriv, max_lotes=1, log=lambda *_: None)
        with self.assertRaises(FileNotFoundError):
            R.ejecutar(self.dir / "no_existe.keras", self.dir / "x.keras", log=lambda *_: None)
        self.assertEqual(R.sha256_archivo(self.orig), self.sha_orig)
        self.assertFalse((self.dir / "x.keras").exists())

    def test_archivo_auxiliar_registra_la_configuracion(self):
        import json
        info = json.loads(self.deriv.with_suffix(".recal.json").read_text(encoding="utf-8"))
        self.assertEqual(info["aplicar_exif"], bool(C.APLICAR_EXIF))
        self.assertEqual(info["origen_sha256"], self.sha_orig)
        self.assertEqual(info["pesos_no_estadistica_cambiados"], 0)

    # ---- 4. reproducibilidad
    def test_reproducible_con_la_misma_semilla_y_distinta_con_otra(self):
        d1, d2, d3 = self.dir / "r1.keras", self.dir / "r2.keras", self.dir / "r3.keras"
        R.ejecutar(self.orig, d1, pasadas=1, max_lotes=LOTES, semilla=7, log=lambda *_: None)
        R.ejecutar(self.orig, d2, pasadas=1, max_lotes=LOTES, semilla=7, log=lambda *_: None)
        R.ejecutar(self.orig, d3, pasadas=1, max_lotes=LOTES, semilla=8, log=lambda *_: None)
        w1, w2, w3 = (R.instantanea(self._cargar(d)) for d in (d1, d2, d3))
        self.assertEqual(R.comparar(w1, w2)["estadisticas_cambiadas"], [])
        self.assertEqual(R.comparar(w1, w2)["pesos_no_estadistica_cambiados"], [])
        self.assertGreater(len(R.comparar(w1, w3)["estadisticas_cambiadas"]), 0)   # la semilla sí influye

    # ---- 5. solo imágenes de train
    def test_usa_exclusivamente_imagenes_de_train(self):
        vistas = []
        original = D.dataset_solo_imagenes
        D.dataset_solo_imagenes = lambda rutas: (vistas.extend(rutas), original(rutas))[1]
        try:
            m = self._cargar(self.orig)
            R.recalibrar(m, R.rutas_train(), pasadas=2, max_lotes=1)
        finally:
            D.dataset_solo_imagenes = original
        self.assertEqual(len(vistas), 2 * len(self.rutas))
        carpeta_train = str((C.DATA_DIR / "train").resolve())
        self.assertTrue(all(str(Path(r).resolve()).startswith(carpeta_train) for r in vistas))
        rutas_val, _ = D.listar_muestras("val")
        self.assertFalse(set(rutas_val) & set(vistas))

    # ---- 6. compatibilidad con EXIF
    def test_la_bandera_exif_llega_a_la_recalibracion(self):
        """Con imágenes EXIF 3/6/8, activar APLICAR_EXIF cambia los píxeles y, por tanto, las estadísticas obtenidas."""
        imgs, _ = D.listar_muestras("train")
        prev = C.APLICAR_EXIF
        try:
            C.APLICAR_EXIF = True
            con = [r for r, o in zip(imgs, D._orientaciones(imgs)) if o in (3, 6, 8)][:16]
        finally:
            C.APLICAR_EXIF = prev
        self.assertEqual(len(con), 16, "no hay suficientes imágenes de train con EXIF 3/6/8")
        res = {}
        for flag in (False, True):
            C.APLICAR_EXIF = flag
            try:
                m = self._cargar(self.orig)
                R.recalibrar(m, con, pasadas=1, max_lotes=1)
                res[flag] = R.instantanea(m)
            finally:
                C.APLICAR_EXIF = prev
        self.assertGreater(len(R.comparar(res[False], res[True])["estadisticas_cambiadas"]), 0)
        # y con la misma bandera es reproducible
        C.APLICAR_EXIF = True
        try:
            m = self._cargar(self.orig)
            R.recalibrar(m, con, pasadas=1, max_lotes=1)
        finally:
            C.APLICAR_EXIF = prev
        self.assertEqual(R.comparar(res[True], R.instantanea(m))["estadisticas_cambiadas"], [])

    def test_el_informe_registra_la_bandera_exif_activa(self):
        d = self.dir / "exif1.keras"
        prev = C.APLICAR_EXIF
        C.APLICAR_EXIF = True
        try:
            info = R.ejecutar(self.orig, d, pasadas=1, max_lotes=1, log=lambda *_: None)
        finally:
            C.APLICAR_EXIF = prev
        self.assertTrue(info["aplicar_exif"])

    # ---- 7. consistencia de la evaluación antes y después
    def test_con_bn_de_lote_las_salidas_no_dependen_de_las_estadisticas_moviles(self):
        """En training=True el BN usa estadísticas del lote: original y derivado deben dar la misma salida."""
        a, b = self._cargar(self.orig), self._cargar(self.deriv)
        ya, yb = a(self.x, training=True), b(self.x, training=True)
        for ta, tb in zip(ya, yb) if isinstance(ya, (list, tuple)) else [(ya, yb)]:
            self.assertTrue(np.allclose(ta.numpy(), tb.numpy(), atol=1e-5))

    def test_en_inferencia_derivado_difiere_del_original(self):
        a, b = self._cargar(self.orig), self._cargar(self.deriv)
        ya, yb = a(self.x, training=False), b(self.x, training=False)
        ta = ya[0] if isinstance(ya, (list, tuple)) else ya
        tb = yb[0] if isinstance(yb, (list, tuple)) else yb
        self.assertFalse(np.allclose(ta.numpy(), tb.numpy(), atol=1e-5))

    def test_guardar_y_recargar_no_altera_las_predicciones(self):
        """La re-guardada del derivado no cambia la inferencia (descarta la 're-guardada' como causa de discrepancias)."""
        b1 = self._cargar(self.deriv)
        re = self.dir / "reguardado.keras"
        b1.save(str(re))
        b2 = self._cargar(re)
        i1, i2 = construir_modelo_inferencia(b1), construir_modelo_inferencia(b2)
        for s1, s2 in zip(i1(self.x, training=False), i2(self.x, training=False)):
            self.assertTrue(np.array_equal(s1.numpy(), s2.numpy()))

    def test_evaluate_predecir_es_determinista_y_no_toca_el_checkpoint(self):
        rutas, _ = D.listar_muestras("val")
        p1 = E.predecir(str(self.deriv), rutas[:8])
        p2 = E.predecir(str(self.deriv), rutas[:8])
        self.assertEqual([len(d) for d in p1], [len(d) for d in p2])
        for d1, d2 in zip(p1, p2):
            for (c1, s1, b1), (c2, s2, b2) in zip(d1, d2):
                self.assertEqual(c1, c2)
                self.assertEqual(s1, s2)
                self.assertTrue(np.array_equal(b1, b2))
        self.assertEqual(R.sha256_archivo(self.orig), self.sha_orig)


@unittest.skipUnless(_hay_train(), "dataset de train no disponible")
class TestLineaDeComandos(unittest.TestCase):
    def test_cli_extremo_a_extremo_y_se_niega_a_sobrescribir(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            origen, salida = t / "o.keras", t / "d.keras"
            construir_ssd().save(str(origen))
            base = [sys.executable, "-B", str(RAIZ / "recalibrar_bn.py"), "--modelo", str(origen), "--salida", str(salida),
                    "--pasadas", "1", "--max-lotes", "1"]
            r = subprocess.run(base, cwd=RAIZ, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            self.assertTrue(salida.exists() and salida.with_suffix(".recal.json").exists())
            r2 = subprocess.run(base, cwd=RAIZ, capture_output=True, text=True)
            self.assertNotEqual(r2.returncode, 0)
            self.assertIn("ya existe", r2.stderr)


if __name__ == "__main__":
    unittest.main()
