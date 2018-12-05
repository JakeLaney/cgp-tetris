from cgp.functions import mathematics as m
from cgp.functions import support as sup
from cgp.functions import lists

import numpy as np
import unittest

# TODO: np.subtract '-' is deprecated for arrays. strange.
# TODO: test function.support.min_shape


class TestList(unittest.TestCase):
    def test_index_y(self):
        x, y = np.random.rand(3, 2), np.random.rand(3, 2)
        real = lists.index_y(x, y, 0)
        print(real)


class TestMath(unittest.TestCase):
    def test_add_int(self):
        x, y = 5, 10
        expected = (x + y) / 2.0
        real = m.add(x, y, 0)
        self.assertEqual(expected, real)

    def test_add_np_array_0(self):
        x, y = np.random.rand(3, 2), np.random.rand(3, 2)
        expected = (x + y) / 2.0
        real = m.add(x, y, 0)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)

    def test_add_np_array_1(self):
        x, y = np.random.rand(2, 3), np.random.rand(3, 2)
        dim = sup.minimum_shape(x, y)
        x, y = np.resize(x, dim), np.resize(y, dim)
        expected = (x + y) / 2.0
        real = m.add(x, y, 0)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)

    def test_aminus_int(self):
        x, y = 10, 5
        expected = 2.5  # np.abs(x - y) / 2.0
        real = m.aminus(x, y, 0)
        self.assertEqual(expected, real)

    def test_aminus_np_array_0(self):
        pass
        # x, y = np.random.rand(3, 2), np.random.rand(3, 2)
        # np.subtract(x, y)
        # expected = np.abs(x - y) / 2.0
        # real = m.aminus(x, y, 0)
        # self.assertEqual(expected, real)

    def test_aminus_np_array_1(self):
        pass
        # x, y = np.random.rand(10, 5), np.random.rand(3, 2)
        # expected = np.abs(x - y) / 2.0
        # real = m.aminus(x, y, 0)
        # self.assertEqual(expected, real)

    def test_mult_int(self):
        x, y = 10, 5
        expected = x * y
        real = m.mult(x, y, 0)
        self.assertEqual(expected, real)

    def test_mult_np_array_0(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        dim = sup.minimum_shape(x, y)
        x, y = np.resize(x, dim), np.resize(y, dim)
        expected = np.multiply(x, y)
        real = m.mult(x, y, 0)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)

    def test_cmult_int(self):
        x, y, p = 10, 5, 100
        expected = x * p
        real = m.cmult(x, y, p)
        self.assertEqual(expected, real)

    def test_cmult_np_array_0(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        p = 10
        dim = sup.minimum_shape(x, y)
        x, y = np.resize(x, dim), np.resize(y, dim)
        expected = x * p
        real = m.cmult(x, y, p)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)

    def test_cmult_np_array_1(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        p = np.random.rand(100, 10)
        dim = sup.minimum_shape(x, p)
        x, p = np.resize(x, dim), np.resize(p, dim)
        expected = x * p
        real = m.cmult(x, y, p)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)

    def test_inv_int_0(self):
        x, y = 1, 0
        expected = 1 / x
        real = m.inv(x, y, 0)
        self.assertEqual(expected, real)

    def test_inv_int_1(self):
        x, y = 0, 0
        expected = 0
        real = m.inv(x, y, 0)
        self.assertEqual(expected, real)

    def test_inv_np_array_0(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        dim = sup.minimum_shape(x, y)
        x, p = np.resize(x, dim), np.resize(y, dim)
        expected = 1 / x
        real = m.inv(x, y, 0)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)

    def test_inv_np_array_1(self):
        x, y = np.zeros(5), np.random.rand(3, 2)
        expected = np.zeros(5)
        real = m.inv(x, y, 0)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)

    def test_abs_int(self):
        x, y = -1, -10
        expected = 1
        real = m.abs(x, y, 0)
        self.assertEqual(expected, real)

    def test_abs_np_array_0(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        dim = sup.minimum_shape(x, y)
        x, p = np.resize(x, dim), np.resize(y, dim)
        expected = np.abs(x)
        real = m.abs(x, y, 0)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)

    def test_sqrt_int(self):
        x, y = -100, -10
        expected = np.sqrt(np.abs(x))
        real = m.sqrt(x, y, 0)
        self.assertEqual(expected, real)

    def test_sqrt_array(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        dim = sup.minimum_shape(x, y)
        x, p = np.resize(x, dim), np.resize(y, dim)
        expected = np.sqrt(x)
        real = m.sqrt(x, y, 0)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)

    def test_cpow_int(self):
        x, y, p = -10, -10, 2
        expected = np.abs(x) ** (p + 1)
        real = m.cpow(x, y, p)
        self.assertEqual(expected, real)

    def test_cpow_array(self):
        x, p, y = np.random.rand(5, 10), np.random.rand(3, 2), 0
        dim = sup.minimum_shape(x, p)
        x, p = np.resize(x, dim), np.resize(p, dim)
        expected = np.abs(x) ** (p + 1)
        real = m.cpow(x, y, p)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)

    def test_ypow_int(self):
        x, y = 10, 510
        expected = np.abs(10) ** np.abs(510)
        real = m.ypow(x, y, 0)
        self.assertEqual(expected, real)

    def test_ypow_array(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        dim = sup.minimum_shape(x, y)
        x, p = np.resize(x, dim), np.resize(y, dim)
        expected = np.abs(x) ** np.abs(y)
        real = m.ypow(x, y, p)
        equal = np.equal(expected, real).all()
        self.assertEqual(equal, True)


