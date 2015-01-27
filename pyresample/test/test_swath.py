from __future__ import with_statement

import os
import sys
import unittest
import warnings
warnings.simplefilter("always")

import numpy as np

from pyresample import kd_tree, geometry


def tmp(f):
    f.tmp = True
    return f


class Test(unittest.TestCase):

    filename = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            'test_files', 'ssmis_swath.npz'))
    data = np.load(filename)['data']
    lons = data[:, 0].astype(np.float64)
    lats = data[:, 1].astype(np.float64)
    tb37v = data[:, 2].astype(np.float64)

    # screen out the fill values
    fvalue = -10000000000.0
    valid_fov = (lons != fvalue) * (lats != fvalue) * (tb37v != fvalue)
    lons = lons[valid_fov]
    lats = lats[valid_fov]
    tb37v = tb37v[valid_fov]

    @tmp
    def test_self_map(self):
        swath_def = geometry.SwathDefinition(lons=self.lons, lats=self.lats)
        if sys.version_info < (2, 6):
            res = kd_tree.resample_gauss(swath_def, self.tb37v.copy(), swath_def,
                                         radius_of_influence=70000, sigmas=56500)
        else:
            with warnings.catch_warnings(record=True) as w:
                res = kd_tree.resample_gauss(swath_def, self.tb37v.copy(), swath_def,
                                             radius_of_influence=70000, sigmas=56500)
                self.assertFalse(
                    len(w) != 1, 'Failed to create neighbour radius warning')
                self.assertFalse(('Possible more' not in str(
                    w[0].message)), 'Failed to create correct neighbour radius warning')
        reffilename = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   'test_files', 'test_self_map.npz'))

        ref = np.load(reffilename)
        self.assertTrue(np.allclose(ref["arr_0"], res),
                        msg='Failed self mapping swath for 1 channel')

    def test_self_map_multi(self):
        data = np.column_stack((self.tb37v, self.tb37v, self.tb37v))
        swath_def = geometry.SwathDefinition(lons=self.lons, lats=self.lats)

        if (sys.version_info < (2, 6) or
                (sys.version_info >= (3, 0) and sys.version_info < (3, 4))):
            res = kd_tree.resample_gauss(swath_def, data, swath_def,
                                         radius_of_influence=70000, sigmas=[56500, 56500, 56500])
        else:
            with warnings.catch_warnings(record=True) as w:
                res = kd_tree.resample_gauss(swath_def, data, swath_def,
                                             radius_of_influence=70000, sigmas=[56500, 56500, 56500])
                self.assertFalse(
                    len(w) != 1, 'Failed to create neighbour radius warning')
                self.assertFalse(('Possible more' not in str(
                    w[0].message)), 'Failed to create correct neighbour radius warning')

        reffilename = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   'test_files', 'test_self_map_multi.npz'))

        ref = np.load(reffilename)["arr_0"]

        self.assertTrue(np.allclose(ref[:, 0], res[:, 0]),
                        msg='Failed self mapping swath multi for 1 channel')

        self.assertTrue(np.allclose(ref[:, 1], res[:, 1]),
                        msg='Failed self mapping swath multi for 1 channel')

        self.assertTrue(np.allclose(ref[:, 2], res[:, 2]),
                        msg='Failed self mapping swath multi for 1 channel')


def suite():
    """The test suite.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test))

    return mysuite
