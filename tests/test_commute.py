#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.commute import commute, main


def _spy(method_to_decorate):
    """Wrap a real method with a MagicMock so calls can be asserted while
    the original implementation still runs."""
    mock = MagicMock(name="groupby method")

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)

    wrapper.mock = mock
    return wrapper


class TestCommute(unittest.TestCase):

    def setUp(self):
        self.df = commute()

    def test_shape(self):
        self.assertEqual(
            self.df.shape,
            (7, 20),
            msg="commute() returned a DataFrame of shape %r, expected "
            "(7, 20) - one row per weekday and one column per counting "
            "station." % (self.df.shape,),
        )

    def test_index(self):
        weekdays = "mon tue wed thu fri sat sun".title().split()
        numbers = list(range(1, 8))
        a = np.all(self.df.index == weekdays)
        b = np.all(self.df.index == numbers)
        self.assertTrue(
            a or b,
            msg="commute() index should be either the weekday names %s or "
            "the weekday numbers %s, got %r."
            % (weekdays, numbers, list(self.df.index)),
        )

    def test_content(self):
        self.assertEqual(
            self.df.values.sum(),
            1264606.0,
            msg="Sum of all elements in the August-2017 weekday totals "
            "should be 1264606.0, got %r." % (self.df.values.sum(),),
        )

    def test_calls(self):
        method = _spy(pd.core.frame.DataFrame.groupby)
        with patch(
            "src.commute.commute", wraps=commute
        ) as pcommute, patch.object(
            pd.core.frame.DataFrame, "groupby", new=method
        ), patch(
            "src.commute.pd.read_csv", wraps=pd.read_csv
        ) as prc, patch(
            "src.commute.plt.show"
        ) as pshow, patch(
            "src.commute.pd.to_datetime", wraps=pd.to_datetime
        ) as pdatetime:
            main()
            pcommute.assert_called_once_with()
            prc.assert_called_once()
            pshow.assert_called_once_with()
            pdatetime.assert_called()
            method.mock.assert_called()


if __name__ == '__main__':
    unittest.main()
