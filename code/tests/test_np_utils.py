import unittest

import numpy as np

import np_utils


class NpUtilsTests(unittest.TestCase):

    def test_expand_images_to_3_channels__3_dimen_case(self):
        images = np.random.rand(5, 128, 128)

        ret = np_utils.expand_images_to_3_channels(images)

        assert isinstance(ret, np.ndarray)
        assert ret.shape == (5, 128, 128, 3)

    def test_expand_images_to_3_channels__4_dimen_case(self):
        images = np.random.rand(5, 128, 128, 1)

        ret = np_utils.expand_images_to_3_channels(images)

        assert isinstance(ret, np.ndarray)
        assert ret.shape == (5, 128, 128, 3)
