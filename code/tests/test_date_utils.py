import datetime
import unittest

import date_utils


class DateUtilsTests(unittest.TestCase):

    def test_timestamp(self):
        dt = datetime.datetime(2017, 1, 1, 12, 1)

        ret = date_utils.timestamp(dt)

        assert ret == "20170101_1201"
