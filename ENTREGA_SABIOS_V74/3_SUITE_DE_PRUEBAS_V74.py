import os
import sys
import unittest
import tempfile
import threading
import time
import socket
import struct
import numpy as np
import jax
import jax.numpy as jnp

DELIVERY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ENTREGA_20260827_V74_"))
if DELIVERY_DIR not in sys.path:
    sys.path.insert(0, DELIVERY_DIR)

import polydim_v74_monolito as pd74

class TestMathCore(unittest.TestCase):
    def test_safe_norm(self):
        x_cero = jnp.zeros(10)
        norm_cero = pd74.safe_norm(x_cero)
        self.assertAlmostEqual(float(jnp.sum(norm_cero)), 0.0)

        x_val = jnp.array([3.0, 4.0])
        self.assertAlmostEqual(float(jnp.sum(pd74.safe_norm(x_val))), 5.0)

        x_complex = jnp.array([1.0 + 1.0j, 1.0 - 1.0j])
        norm_complex = pd74.safe_norm(x_complex)
        self.assertEqual(norm_complex.dtype, jnp.complex64)  
        self.assertAlmostEqual(float(jnp.real(jnp.sum(norm_complex))), 2.0, places=5)

    def test_safe_dot(self):
        x = jnp.array([1.0, 0.0])
        y = jnp.array([0.0, 1.0])
        self.assertAlmostEqual(float(jnp.sum(pd74.safe_dot(x, y))), 0.0)
        
        x_int = jnp.array([1, 2], dtype=jnp.int32)
        y_int = jnp.array([2, 1], dtype=jnp.int32)
        res = pd74.safe_dot(x_int, y_int)
        self.assertTrue(jnp.issubdtype(res.dtype, jnp.floating))

class TestGeodesicKernels(unittest.TestCase):
    def test_exp_map_and_log_map(self):
        x = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, jnp.pi/2, 0.0])
        y = pd74.GeodesicKernels.exp_map(x, v)
        self.assertTrue(jnp.allclose(y, jnp.array([0.0, 1.0, 0.0]), atol=1e-5))
        v_recovered = pd74.GeodesicKernels.log_map(x, y)
        self.assertTrue(jnp.allclose(v, v_recovered, atol=1e-5))

    def test_log_map_antipodal(self):
        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([-1.0, 0.0, 0.0])
        v = pd74.GeodesicKernels.log_map(x, y)
        norm_v = pd74.safe_norm(v)
        self.assertAlmostEqual(float(jnp.sum(norm_v)), jnp.pi, places=5)

    def test_log_map_newton(self):
        x = jnp.array([0.0, 1.0])
        y = jnp.array([1.0, 0.0])
        v = pd74.GeodesicKernels.log_map_newton(x, y)
        y_hat = pd74.GeodesicKernels.exp_map(x, v)
        self.assertTrue(jnp.allclose(y, y_hat, atol=1e-5))

    def test_slerp(self):
        x = jnp.array([1.0, 0.0])
        y = jnp.array([0.0, 1.0])
        mid = pd74.GeodesicKernels.slerp(x, y, 0.5)
        expected = jnp.array([jnp.sqrt(2)/2, jnp.sqrt(2)/2])
        self.assertTrue(jnp.allclose(mid, expected, atol=1e-5))

class TestCayleyAndClifford(unittest.TestCase):
    def test_cayley_transform(self):
        A = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
        # En la V74, cayley_transform est dentro de CliffordRotors
        Q = pd74.CliffordRotors.cayley_transform(A)
        I = jnp.eye(4)
        self.assertTrue(jnp.allclose(Q @ Q.T, I, atol=1e-4))

    def test_apply_spherical_rotor(self):
        U = jnp.array([1.0, 0.0, 0.0])
        V = jnp.array([0.0, 1.0, 0.0])
        x = jnp.array([1.0, 0.0, 0.0])
        x_rot = pd74.CliffordRotors.apply_spherical_rotor(x, U, V)
        # Rotar x preserva la norma
        self.assertAlmostEqual(float(jnp.sum(pd74.safe_norm(x_rot))), 1.0, places=5)

class TestFFIBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pd74.NativeFFIBridge.initialize()

    @classmethod
    def tearDownClass(cls):
        pd74.NativeFFIBridge.cleanup()

    def test_householder_reflect(self):
        x = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([1.0, 1.0, 0.0])
        out = pd74.NativeFFIBridge.householder_reflect(x, v)
        self.assertTrue(jnp.allclose(out, jnp.array([0.0, -1.0, 0.0]), atol=1e-6))

    def test_scrub_subnormals(self):
        f32_sub = struct.unpack('<f', struct.pack('<i', 1))[0]
        t = jnp.array([f32_sub, 1.0], dtype=jnp.float32)
        t_clean = pd74.NativeFFIBridge.scrub_subnormals(t)
        self.assertEqual(float(t_clean[0]), 0.0)
        self.assertEqual(float(t_clean[1]), 1.0)

class TestStorage(unittest.TestCase):
    def test_pmtp_storage_roundtrip(self):
        t_orig = jax.random.normal(jax.random.PRNGKey(42), (100, 100))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "test.pmtp")
            pd74.PMTPPersistentStorage.save_tensor(path, t_orig).result()
            meta = pd74.PMTPPersistentStorage.read_metadata(path)
            self.assertEqual(tuple(meta["shape"]), (100, 100))
            t_load = pd74.PMTPPersistentStorage.load_tensor(path)
            self.assertTrue(jnp.array_equal(t_orig, t_load))

class TestNetwork(unittest.TestCase):
    def test_pmtp_network_roundtrip(self):
        """Prueba de transmisión TCP con protocolo PMTP."""
        bridge = pd74.PMTPAgentBridge(host="127.0.0.1", port=0)
        bridge.start_server()
        port = bridge.port
        
        t_orig = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        
        # Enviar
        success = pd74.PMTPAgentBridge.send_tensor("127.0.0.1", port, t_orig)
        self.assertTrue(success, "El envío falló")
        
        # Recibir
        t_recv = bridge.inbox.get(timeout=2.0)
        self.assertTrue(jnp.array_equal(t_orig, t_recv))
        
        bridge.stop_server()

if __name__ == "__main__":
    print("=" * 60)
    print(" SUITE DE PRUEBAS COMPLETA (ELEMENTO POR ELEMENTO) V74")
    print("=" * 60)
    unittest.main(verbosity=2)
