import sys
from os import getcwd
sys.path.append(getcwd()) # if run from root
sys.path.append(getcwd() + '/..') # if run from test/

from cgp.functions import mathematics as mat
from cgp.functions import support as supp
from cgp.functions import lists

import numpy as np
import unittest

# TODO: np.subtract '-' is deprecated for arrays. strange.
# TODO: test function.support.min_shape

class TestSupport(unittest.TestCase):
    def test_pass_through(self):
        inp = 1
        exp = True
        act = supp.is_scalar(inp)
        self.assertEqual(exp, act)
        inp = np.array([1])
        exp = False
        act = supp.is_scalar(inp)
        self.assertEqual(exp, act)
        inp = np.array([[1],[2]])
        exp = False
        act = supp.is_scalar(inp)
        self.assertEqual(exp, act)
        inp = np.array([[1],[2]])
        exp = False
        act = supp.is_scalar(inp)
        self.assertEqual(exp, act)
        inp = np.array([1, 2, 3])
        exp = False
        act = supp.is_scalar(inp)
        self.assertEqual(exp, act)

    def test_min_dim(self):
        a = np.array([1])
        b = np.array([1, 2])
        exp = [1]
        act = supp.min_dim(a, b)
        self.assertListEqual(exp, act)

        a = np.array([[1], [1]])
        b = np.array([1, 2])
        exp = [2]
        act = supp.min_dim(a, b)
        self.assertListEqual(exp, act)

class TestList(unittest.TestCase):
    def test_split_before(self):
        inp = 1
        exp = 1
        act = lists.split_before(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1])
        exp = 1
        act = lists.split_before(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1, 2, 3])
        exp = np.array([1])
        act = lists.split_before(inp, 0, -1.0)
        self.assertTrue(np.array_equal(exp, act))
        inp = np.array([1, 2, 3])
        exp = np.array([1, 2, 3])
        act = lists.split_before(inp, 0, 1.0)
        self.assertTrue(np.array_equal(exp, act))

    def test_split_after(self):
        inp = 1
        exp = 1
        act = lists.split_after(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1])
        exp = 1
        act = lists.split_after(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1, 2, 3])
        exp = np.array([1, 2, 3])
        act = lists.split_after(inp, 0, -1.0)
        self.assertTrue(np.array_equal(exp, act))
        inp = np.array([1, 2, 3])
        exp = np.array([3])
        act = lists.split_after(inp, 0, 1.0)
        self.assertTrue(np.array_equal(exp, act))

    def test_range_in(self):
        inp = 1
        exp = 1
        act = lists.range_in(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1])
        exp = 1
        act = lists.range_in(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1, 2, 3])
        exp = np.array([2])
        act = lists.range_in(inp, 0, 0)
        self.assertTrue(np.array_equal(exp, act))
        inp = np.array([1, 2, 3])
        exp = np.array([1, 2, 3])
        act = lists.range_in(inp, -1.0, 1.0)
        self.assertTrue(np.array_equal(exp, act))
        inp = np.array([1, 2, 3])
        exp = np.array([1, 2, 3])
        act = lists.range_in(inp, 1, -1)
        self.assertTrue(np.array_equal(exp, act))

    def test_index_y(self):
        inp = 1
        exp = 1
        act = lists.index_y(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1])
        exp = 1
        act = lists.index_y(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1, 2, 3])
        exp = 1
        act = lists.index_y(inp, -1.0, 0)
        self.assertTrue(exp == act)
        inp = np.array([1, 2, 3])
        exp = 3
        act = lists.index_y(inp, 1.0, 1)
        self.assertTrue(exp == act)

    def test_index_p(self):
        inp = 1
        exp = 1
        act = lists.index_p(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1])
        exp = 1
        act = lists.index_p(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1, 2, 3])
        exp = 1
        act = lists.index_p(inp, -1.0, -1.0)
        self.assertTrue(exp == act)
        inp = np.array([1, 2, 3])
        exp = 3
        act = lists.index_p(inp, 1.0, 1.0)
        self.assertTrue(exp == act)

    def test_vectorize(self):
        inp = 1
        exp = 1
        act = lists.vectorize(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([1])
        exp = 1
        act = lists.vectorize(inp, 0, 0)
        self.assertEqual(exp, act)
        inp = np.array([[1, 2, 3], [4, 5, 6]])
        exp = [1, 2, 3, 4, 5, 6]
        act = lists.vectorize(inp, -1.0, -1.0)
        self.assertTrue(np.array_equal(exp, act))

    def test_f_first(self):
        inp = np.array([[1, 2, 3], [4, 5, 6]])
        exp = 1
        act = lists.f_first(inp, -1.0, -1.0)
        self.assertTrue(exp, act)

    def test_f_last(self):
        inp = np.array([[1, 2, 3], [4, 5, 6]])
        exp = 1
        act = lists.f_last(inp, -1.0, -1.0)
        self.assertTrue(exp, act)

    def test_differences(self):
        inp = np.array([[1, 2, 3], [4, 5, 6]])
        exp = [1, 1, 1, 1, 1]
        act = lists.differences(inp, -1.0, -1.0)
        self.assertTrue(np.array_equal(exp, act))

    def test_push_back(self):
        x = 1
        y = -1
        exp = [1, -1]
        act = lists.push_back(x, y, -1.0)
        self.assertTrue(np.array_equal(exp, act))
        x = 1
        y = [-1, 2]
        exp = [1, -1, 2]
        act = lists.push_back(x, y, -1.0)
        self.assertTrue(np.array_equal(exp, act))

    def test_set_x(self):
        x = 5
        y = [1, 3, 4]
        exp = [5, 5, 5]
        act = lists.set_x(x, y, -1.0)
        self.assertTrue(np.array_equal(exp, act))

    def test_vec_from_double(self):
        x = 5
        y = [1, 3, 4]
        exp = [5]
        act = lists.vec_from_double(x, y, -1.0)
        self.assertTrue(np.array_equal(exp, act))

    def test_constvectord(self):
        x = 5
        y = [1, 3, 4]
        p = 0.1
        exp = [0.1]
        act = lists.constvectord(x, y, p)
        self.assertTrue(np.array_equal(exp, act))
        x = np.array([[1,2],[3,4]])
        y = [1, 3, 4]
        p = 0.1
        exp = [[0.1,0.1],[0.1,0.1]]
        act = lists.constvectord(x, y, p)
        self.assertTrue(np.array_equal(exp, act))

    def test_zeros(self):
        x = 5
        y = [1, 3, 4]
        p = 0.1
        exp = [0]
        act = lists.zeros(x, y, p)
        self.assertTrue(np.array_equal(exp, act))
        x = np.array([[1,2],[3,4]])
        y = [1, 3, 4]
        p = 0.1
        exp = [[0,0],[0,0]]
        act = lists.zeros(x, y, p)
        self.assertTrue(np.array_equal(exp, act))

    def test_ones(self):
        x = 5
        y = [1, 3, 4]
        p = 0.1
        exp = [1]
        act = lists.ones(x, y, p)
        self.assertTrue(np.array_equal(exp, act))
        x = np.array([[1,2],[3,4]])
        y = [1, 3, 4]
        p = 0.1
        exp = [[1,1],[1,1]]
        act = lists.ones(x, y, p)
        self.assertTrue(np.array_equal(exp, act))

class TestMath(unittest.TestCase):
    def test_add_int(self):
        x, y = 5, 10
        exp = (x + y) / 2.0
        act = mat.add(x, y, 0)
        self.assertEqual(exp, act)

    def test_add_np_array_0(self):
        x, y = np.random.rand(3, 2), np.random.rand(3, 2)
        exp = (x + y) / 2.0
        act = mat.add(x, y, 0)
        self.assertTrue(np.array_equal(exp, act))

    def test_add_np_array_1(self):
        x, y = np.random.rand(2, 3), np.random.rand(3, 2)
        dim = supp.min_dim(x, y)
        x, y = np.resize(x, dim), np.resize(y, dim)
        exp = (x + y) / 2.0
        act = mat.add(x, y, 0)
        equal = np.equal(exp, act).all()
        self.assertEqual(equal, True)

    def test_aminus_int(self):
        x, y = 10, 5
        exp = 2.5  # np.abs(x - y) / 2.0
        act = mat.aminus(x, y, 0)
        self.assertEqual(exp, act)

    def test_aminus_np_array_0(self):
        pass
        # x, y = np.random.rand(3, 2), np.random.rand(3, 2)
        # np.subtract(x, y)
        # exp = np.abs(x - y) / 2.0
        # act = mat.aminus(x, y, 0)
        # self.assertEqual(exp, act)

    def test_aminus_np_array_1(self):
        pass
        # x, y = np.random.rand(10, 5), np.random.rand(3, 2)
        # exp = np.abs(x - y) / 2.0
        # act = mat.aminus(x, y, 0)
        # self.assertEqual(exp, act)

    def test_mult_int(self):
        x, y = 10, 5
        exp = x * y
        act = mat.mult(x, y, 0)
        self.assertEqual(exp, act)

    def test_mult_np_array_0(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        dim = supp.min_dim(x, y)
        x, y = np.resize(x, dim), np.resize(y, dim)
        exp = np.multiply(x, y)
        act = mat.mult(x, y, 0)
        equal = np.equal(exp, act).all()
        self.assertEqual(equal, True)

    def test_cmult_int(self):
        x, y, p = 10, 5, 100
        exp = x * p
        act = mat.cmult(x, y, p)
        self.assertEqual(exp, act)

    def test_cmult_np_array_0(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        p = 10
        dim = supp.min_dim(x, y)
        x, y = np.resize(x, dim), np.resize(y, dim)
        exp = x * p
        act = mat.cmult(x, y, p)
        equal = np.equal(exp, act).all()
        self.assertEqual(equal, True)

    def test_cmult_np_array_1(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        p = np.random.rand(100, 10)
        dim = supp.min_dim(x, p)
        x, p = np.resize(x, dim), np.resize(p, dim)
        exp = x * p
        act = mat.cmult(x, y, p)
        equal = np.equal(exp, act).all()
        self.assertEqual(equal, True)

    def test_inv_int_0(self):
        x, y = 1, 0
        exp = 1 / x
        act = mat.inv(x, y, 0)
        self.assertEqual(exp, act)

    def test_inv_int_1(self):
        x, y = 0, 0
        exp = 0
        act = mat.inv(x, y, 0)
        self.assertEqual(exp, act)

    def test_inv_np_array_0(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        dim = supp.min_dim(x, y)
        x, p = np.resize(x, dim), np.resize(y, dim)
        exp = 1 / x
        act = mat.inv(x, y, 0)
        equal = np.equal(exp, act).all()
        self.assertEqual(equal, True)

    def test_inv_np_array_1(self):
        x = np.zeros(5)
        exp = np.zeros(5)
        act = mat.inv(x, 0, 0)
        self.assertTrue(np.array_equal(exp, act))

    def test_abs_int(self):
        x, y = -1, -10
        exp = 1
        act = mat.abs(x, y, 0)
        self.assertEqual(exp, act)

    def test_abs_np_array_0(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        dim = supp.min_dim(x, y)
        x, p = np.resize(x, dim), np.resize(y, dim)
        exp = np.abs(x)
        act = mat.abs(x, y, 0)
        equal = np.equal(exp, act).all()
        self.assertEqual(equal, True)

    def test_sqrt_int(self):
        x, y = -100, -10
        exp = np.sqrt(np.abs(x))
        act = mat.sqrt(x, y, 0)
        self.assertEqual(exp, act)

    def test_sqrt_array(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        dim = supp.min_dim(x, y)
        x, p = np.resize(x, dim), np.resize(y, dim)
        exp = np.sqrt(x)
        act = mat.sqrt(x, y, 0)
        equal = np.equal(exp, act).all()
        self.assertEqual(equal, True)

    def test_cpow_int(self):
        x, y, p = -10, -10, 2
        exp = np.abs(x) ** (p + 1)
        act = mat.cpow(x, y, p)
        self.assertEqual(exp, act)

    def test_cpow_array(self):
        x, p, y = np.random.rand(5, 10), np.random.rand(3, 2), 0
        dim = supp.min_dim(x, p)
        x, p = np.resize(x, dim), np.resize(p, dim)
        exp = np.abs(x) ** (p + 1)
        act = mat.cpow(x, y, p)
        equal = np.equal(exp, act).all()
        self.assertEqual(equal, True)

    def test_ypow_int(self):
        x, y = 10, 510
        exp = np.abs(10) ** np.abs(510)
        act = mat.ypow(x, y, 0)
        self.assertEqual(exp, act)

    def test_ypow_array(self):
        x, y = np.random.rand(5, 10), np.random.rand(3, 2)
        dim = supp.min_dim(x, y)
        x, p = np.resize(x, dim), np.resize(y, dim)
        exp = np.abs(x) ** np.abs(y)
        act = mat.ypow(x, y, p)
        equal = np.equal(exp, act).all()
        self.assertEqual(equal, True)

if __name__ == '__main__':
    unittest.main()
