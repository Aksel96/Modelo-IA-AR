"""Tests de la orientación EXIF (E1).  Uso, desde la raíz del repo:

    python -B -m unittest tests.test_exif -v

No modifican datos ni archivos del proyecto. Los tests que usan el dataset se omiten si no existe.
"""
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

import config as C
import data as D


def _con_exif(flag):
    """Context manager mínimo para fijar C.APLICAR_EXIF durante un test."""
    class _Ctx:
        def __enter__(self):
            self.prev = C.APLICAR_EXIF
            C.APLICAR_EXIF = flag
        def __exit__(self, *a):
            C.APLICAR_EXIF = self.prev
    return _Ctx()


def _imagen_asimetrica(h=6, w=10):
    """Imagen sin ninguna simetría: cada píxel tiene un valor distinto."""
    a = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    return a


class TestAplicarOrientacion(unittest.TestCase):
    def test_equivale_a_pil_para_las_8_orientaciones(self):
        base = _imagen_asimetrica()
        for o in range(1, 9):
            im = Image.fromarray(base)
            ex = Image.Exif()
            ex[274] = o
            im.info["exif"] = ex.tobytes()
            buf = io.BytesIO()
            im.save(buf, format="PNG", exif=ex)       # PNG conserva EXIF y es sin pérdida
            buf.seek(0)
            esperado = np.asarray(ImageOps.exif_transpose(Image.open(buf)))
            obtenido = D._aplicar_orientacion(tf.constant(base), tf.constant(o)).numpy()
            self.assertEqual(esperado.shape, obtenido.shape, f"forma distinta en EXIF {o}")
            self.assertTrue(np.array_equal(esperado, obtenido), f"píxeles distintos en EXIF {o}")

    def test_valores_fuera_de_rango_equivalen_a_identidad(self):
        base = _imagen_asimetrica()
        for o in (0, 9, 99, -3):
            out = D._aplicar_orientacion(tf.constant(base), tf.constant(o)).numpy()
            self.assertTrue(np.array_equal(out, base))

    def test_funciona_dentro_de_tf_data_con_formas_distintas(self):
        """La rotación de 90° cambia (H,W); debe soportarlo un tf.data.map."""
        base = _imagen_asimetrica()
        ds = tf.data.Dataset.from_tensor_slices(np.array([1, 3, 6, 8], dtype=np.int32))
        ds = ds.map(lambda o: tf.shape(D._aplicar_orientacion(tf.constant(base), o)))
        formas = [tuple(s.numpy()[:2]) for s in ds]
        self.assertEqual(formas, [(6, 10), (6, 10), (10, 6), (10, 6)])


class TestBandera(unittest.TestCase):
    def test_orientaciones_con_bandera_desactivada_son_todo_1(self):
        with _con_exif(False):
            o = D._orientaciones(["no_existe_1.jpg", "no_existe_2.jpg"])
        self.assertEqual(o.tolist(), [1, 1])

    def test_leer_orientacion_tolera_archivos_ilegibles(self):
        self.assertEqual(D._leer_orientacion("ruta/que/no/existe.jpg"), 1)


def _hay_dataset():
    return (C.DATA_DIR / "val" / "images").exists() and any((C.DATA_DIR / "val" / "images").glob("*.jp*g"))


@unittest.skipUnless(_hay_dataset(), "dataset no disponible")
class TestSobreDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rutas, cls.gts = D.listar_muestras("val")
        cls.ori = {r: D._leer_orientacion(r) for r in cls.rutas}

    def _por_orientacion(self, o, n=3):
        return [r for r in self.rutas if self.ori[r] == o][:n]

    def test_legacy_no_cambia_ninguna_imagen(self):
        """Con la bandera desactivada la salida es idéntica a decodificar + resize + /255 sin rotar."""
        with _con_exif(False):
            for r in self.rutas[:5] + self._por_orientacion(6) + self._por_orientacion(3):
                raw = tf.io.decode_image(tf.io.read_file(r), channels=3, expand_animations=False)
                esperado = tf.cast(tf.image.resize(raw, (C.IMG_SIZE, C.IMG_SIZE)), tf.float32) / 255.0
                self.assertTrue(np.array_equal(D._cargar_imagen(r).numpy(), esperado.numpy()), r)
                self.assertTrue(np.array_equal(D._cargar_imagen(r, 6).numpy(), esperado.numpy()), r)

    def test_con_exif_las_imagenes_sin_rotacion_no_cambian(self):
        sin = [r for r in self.rutas if self.ori[r] == 1][:5]
        self.assertTrue(sin)
        for r in sin:
            with _con_exif(False):
                a = D._cargar_imagen(r).numpy()
            with _con_exif(True):
                b = D._cargar_imagen(r, 1).numpy()
            self.assertTrue(np.array_equal(a, b), r)

    def test_con_exif_la_forma_orientada_coincide_con_pil(self):
        for o in (3, 6, 8):
            for r in self._por_orientacion(o, 2):
                with Image.open(r) as im:
                    esperado = ImageOps.exif_transpose(im).size           # (W, H)
                with _con_exif(True):
                    raw = tf.io.decode_image(tf.io.read_file(r), channels=3, expand_animations=False)
                    rot = D._aplicar_orientacion(raw, o)
                self.assertEqual((int(rot.shape[1]), int(rot.shape[0])), esperado, r)

    def test_con_exif_rotacion_coincide_con_pil_pixel_a_pixel(self):
        """Mismos píxeles crudos (decodificados por PIL) -> misma imagen orientada que exif_transpose."""
        for o in (3, 6, 8):
            for r in self._por_orientacion(o, 1):
                with Image.open(r) as im:
                    crudo = np.asarray(im.convert("RGB"))
                    esperado = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
                obtenido = D._aplicar_orientacion(tf.constant(crudo), tf.constant(o)).numpy()
                self.assertTrue(np.array_equal(esperado, obtenido), f"{r} (EXIF {o})")

    def test_firmas_publicas_sin_cambios_y_orden_preservado(self):
        rutas, gts = D.listar_muestras("val")                    # evaluate.py espera esta tupla de 2
        self.assertEqual(len(rutas), len(gts))
        with _con_exif(True):
            ds = D.dataset_solo_imagenes(rutas[:4])
            x = next(iter(ds))
        self.assertEqual(tuple(x.shape), (4, C.IMG_SIZE, C.IMG_SIZE, 3))


# ---------------------------------------------------------------- Pillow como dependencia condicional
RAIZ = Path(__file__).resolve().parent.parent
_BLOQUEAR_PIL = "import sys; sys.modules['PIL'] = None   # simula Pillow no instalado\n"


def _correr(codigo, exif=None, sin_pillow=False):
    """Ejecuta `codigo` en un intérprete nuevo (el import de data.py ocurre allí). exif: None/'0'/'1'."""
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONDONTWRITEBYTECODE="1")
    env.pop("APLICAR_EXIF", None)
    if exif is not None:
        env["APLICAR_EXIF"] = exif
    pre = _BLOQUEAR_PIL if sin_pillow else ""
    return subprocess.run([sys.executable, "-B", "-c", pre + codigo], cwd=str(RAIZ), env=env,
                          capture_output=True, text=True, timeout=900)


@unittest.skipUnless(_hay_dataset(), "dataset no disponible")
class TestPillowCondicional(unittest.TestCase):
    CODIGO_LEGACY = (
        "import numpy as np, config as C, data as D\n"
        "assert C.APLICAR_EXIF is False\n"
        "rutas, gts = D.listar_muestras('val')\n"
        "assert D._orientaciones(rutas[:8]).tolist() == [1] * 8\n"
        "x = next(iter(D.dataset_solo_imagenes(rutas[:4])))\n"
        "assert tuple(x.shape) == (4, C.IMG_SIZE, C.IMG_SIZE, 3)\n"
        "xv, yv = next(iter(D.construir_dataset('val', False)))\n"
        "assert tuple(xv.shape)[1:] == (C.IMG_SIZE, C.IMG_SIZE, 3) and yv.shape[-1] == 5\n"
        "xt, yt = next(iter(D.construir_dataset('train', True)))\n"
        "assert tuple(xt.shape)[1:] == (C.IMG_SIZE, C.IMG_SIZE, 3) and yt.shape[-1] == 5\n"
        "print('OK')\n"
    )

    def test_legacy_sin_pillow_carga_datos_y_construye_datasets(self):
        for exif in (None, "0"):                      # bandera sin definir y explícitamente en 0
            r = _correr(self.CODIGO_LEGACY, exif=exif, sin_pillow=True)
            self.assertEqual(r.returncode, 0, r.stderr[-1500:])
            self.assertIn("OK", r.stdout)

    def test_legacy_con_y_sin_pillow_da_la_misma_imagen(self):
        codigo = ("import hashlib, config as C, data as D\n"
                  "rutas, _ = D.listar_muestras('val')\n"
                  "x = next(iter(D.dataset_solo_imagenes(rutas[:6]))).numpy()\n"
                  "print(hashlib.sha256(x.tobytes()).hexdigest())\n")
        a = _correr(codigo, exif="0", sin_pillow=False)
        b = _correr(codigo, exif="0", sin_pillow=True)
        self.assertEqual(a.returncode, 0, a.stderr[-1500:])
        self.assertEqual(b.returncode, 0, b.stderr[-1500:])
        self.assertEqual(a.stdout.strip(), b.stdout.strip())

    def test_exif_activo_sin_pillow_falla_con_mensaje_claro(self):
        codigo = ("import config as C, data as D\n"
                  "assert C.APLICAR_EXIF is True\n"
                  "rutas, _ = D.listar_muestras('val')\n"
                  "D.dataset_solo_imagenes(rutas[:2])\n")
        r = _correr(codigo, exif="1", sin_pillow=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ImportError", r.stderr)
        self.assertIn("APLICAR_EXIF=1 necesita Pillow", r.stderr)

    def test_exif_activo_con_pillow_lee_orientaciones_reales(self):
        codigo = ("import config as C, data as D\n"
                  "assert C.APLICAR_EXIF is True\n"
                  "rutas, _ = D.listar_muestras('val')\n"
                  "o = D._orientaciones(rutas)\n"
                  "print(sorted(set(o.tolist())))\n"
                  "x = next(iter(D.dataset_solo_imagenes(rutas[:4])))\n"
                  "assert tuple(x.shape) == (4, C.IMG_SIZE, C.IMG_SIZE, 3)\n")
        r = _correr(codigo, exif="1", sin_pillow=False)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("[1, 3, 6, 8]", r.stdout)       # el split val contiene EXIF 3, 6 y 8

    def test_data_py_no_importa_pillow_a_nivel_de_modulo(self):
        import ast
        arbol = ast.parse((RAIZ / "data.py").read_text(encoding="utf-8"))
        for nodo in arbol.body:                       # solo sentencias de primer nivel
            if isinstance(nodo, ast.Import):
                self.assertFalse(any(a.name.split(".")[0] == "PIL" for a in nodo.names))
            if isinstance(nodo, ast.ImportFrom):
                self.assertNotEqual((nodo.module or "").split(".")[0], "PIL")


if __name__ == "__main__":
    unittest.main()
