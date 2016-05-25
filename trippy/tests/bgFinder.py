# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals, print_function
from mock import (MagicMock, patch, )
from unittest import TestCase
from numpy import array
from trippy.bgFinder import bgFinder


class TestBGFinder(TestCase):

    @staticmethod
    def create_finder():
        retval = bgFinder(MagicMock())
        retval.data = MagicMock()
        return retval

    @patch('trippy.bgFinder.num.median')
    def test_call_default(self, mock_median):
        'Test the default values of the ``__call__`` method'
        b = self.create_finder()
        r = b()

        mock_median.assert_called_once_with(b.data)
        self.assertEqual(mock_median(), r)

    @patch('trippy.bgFinder.num.median')
    def test_call_median(self, mock_median):
        'Test the ``median`` parameter of the ``__call__`` method'
        b = self.create_finder()
        r = b('median')

        mock_median.assert_called_once_with(b.data)
        self.assertEqual(mock_median(), r)

    @patch('trippy.bgFinder.num.mean')
    def test_call_mean(self, mock_mean):
        'Test the ``mean`` parameter of the ``__call__`` method'
        b = self.create_finder()
        r = b('mean')

        mock_mean.assert_called_once_with(b.data)
        self.assertEqual(mock_mean(), r)

    @patch.object(bgFinder, 'histMode')
    def test_call_histMode(self, mock_histMode):
        'Test the ``histMode`` parameter of the ``__call__`` method'
        b = self.create_finder()
        mock_histMode.return_value = "I'm a lumberjack"
        r = b('histMode')

        mock_histMode.assert_called_once_with()
        self.assertEqual("I'm a lumberjack", r)

    @patch.object(bgFinder, 'histMode')
    def test_call_histMode_imp(self, mock_histMode):
        'Test the ``histMode`` parameter of the ``__call__`` method with ``inp`` set'
        b = self.create_finder()
        mock_histMode.return_value = "I'm a lumberjack"
        r = b('histMode', 9)

        mock_histMode.assert_called_once_with(9)
        self.assertEqual("I'm a lumberjack", r)

    @patch.object(bgFinder, 'fraserMode')
    def test_call_fraser(self, mock_fraser):
        'Test the ``fraserMode`` parameter of the ``__call__`` method'
        b = self.create_finder()
        mock_fraser.return_value = "I'm a lumberjack"
        r = b('fraserMode')

        mock_fraser.assert_called_once_with()
        self.assertEqual(mock_fraser(), r)

    @patch.object(bgFinder, 'fraserMode')
    def test_call_fraserInp(self, mock_fraser):
        'Test the ``fraserMode`` parameter of the ``__call__`` method with ``imp`` set'
        b = self.create_finder()
        mock_fraser.return_value = "I'm a lumberjack"
        r = b('fraserMode', 0.9)

        mock_fraser.assert_called_once_with(0.9)
        self.assertEqual(mock_fraser(), r)

    @patch.object(bgFinder, '_gaussFit')
    def test_call_gauss(self, mock_gauss):
        'Test the ``gaussFit`` parameter of the ``__call__`` method'
        b = self.create_finder()
        mock_gauss.return_value = "I'm a lumberjack"
        r = b('gaussFit')

        mock_gauss.assert_called_once_with()
        self.assertEqual(mock_gauss(), r)

    @patch.object(bgFinder, 'smartBackground')
    def test_call_smart(self, mock_smart):
        'Test the ``smart`` parameter of the ``__call__`` method'
        b = self.create_finder()
        mock_smart.return_value = "I'm a lumberjack"
        r = b('smart')

        mock_smart.assert_called_once_with()
        self.assertEqual(mock_smart(), r)

    def test_call_unknown(self):
        'Ensure we raise an error if a weird method is passed in'
        b = self.create_finder()
        with self.assertRaises(ValueError):
            b("I'm a lumberjack'")

    @patch.object(bgFinder, '_stats')
    def test_histMode(self, mock_stats):
        'Test the default arguments to the ``bgFinder.histMode`` method'
        mock_stats.return_value = ["I'm a lumberjack"]
        b = self.create_finder()
        r = b.histMode()

        mock_stats.assert_called_once_with(50)
        self.assertEqual(mock_stats()[0], r)

    @patch.object(bgFinder, '_stats')
    def test_histMode_val(self, mock_stats):
        'Test the ``bgFinder.histMode`` method with the parameter set'
        b = self.create_finder()
        mock_stats.return_value = ["I'm a lumberjack"]
        r = b.histMode(9)

        mock_stats.assert_called_once_with(9)
        self.assertEqual(mock_stats()[0], r)

    @patch.object(bgFinder, '_fraserMode')
    def test_fraserMode(self, mock_fraser):
        'Test the ``bgFinder.fraserMode`` method '
        b = self.create_finder()
        mock_fraser.return_value = "I'm a lumberjack"
        r = b.fraserMode()

        mock_fraser.assert_called_once_with(0.1)
        self.assertEqual(mock_fraser(), r)

    @patch.object(bgFinder, '_fraserMode')
    def test_fraserMode_val(self, mock_fraser):
        'Test the ``bgFinder.fraserMode`` method with the parameter set'
        b = self.create_finder()
        mock_fraser.return_value = "I'm a lumberjack"
        r = b.fraserMode(0.2)

        mock_fraser.assert_called_once_with(0.2)
        self.assertEqual(mock_fraser(), r)

    def test_ahist(self):
        b = self.create_finder()
        data = array(range(0, 255), int)
        r = b._ahist(data, 50)

        self.assertEqual(50, len(r[0]))
        self.assertEqual(5, r[1])
        self.assertEqual(3, r[2])

    def test_stats(self):
        data = array(range(0, 255), int)
        b = bgFinder(data)
        r = b._stats()

        self.assertEqual(8, r[0])
        self.assertAlmostEqual(85.1, r[1], places=1)

    def test_under_fraserMode(self):
        'Test ``bgFinder._fraserMode``'
        data = array(range(0, 255), int)
        b = bgFinder(data)
        r = b._fraserMode()

        self.assertAlmostEqual(4.5, r, places=1)

    def test_gaussFit(self):
        data = array(range(0, 255), int)
        b = bgFinder(data)
        r = b._gaussFit()

        self.assertAlmostEqual(127.0, r, places=1)
        self.assertAlmostEqual(r, b.gauss[0], places=1)
        self.assertAlmostEqual(73.61159329, b.gauss[1], places=1)

    def test_gaussLike(self):
        data = array(range(0, 255), int)
        b = bgFinder(data)
        r = b._gaussLike([1.0, 2.0])

        self.assertAlmostEqual(679178.581857, r, places=1)

    @patch.object(bgFinder, '_gaussFit')
    def test_smartBackground_high_limit(self, mock_gaussFit):
        g = (1, 0.2)  # The Gauss value
        mock_gaussFit.return_value = g
        b = self.create_finder()
        b.gauss = g
        r = b.smartBackground()

        self.assertEqual(g[0], r)

    @patch.object(bgFinder, '_gaussFit')
    @patch.object(bgFinder, '__call__')
    def test_smartBackground_low_limit_inp_none(self, mock_call, mock_gaussFit):
        g = (1, 2)  # The Gauss value
        mock_gaussFit.return_value = g
        mock_call.return_value = 2

        b = self.create_finder()
        b.gauss = g
        r = b.smartBackground()

        self.assertEqual(r, 2)
        mock_call.assert_called_once_with('fraserMode')

    @patch.object(bgFinder, '_gaussFit')
    @patch.object(bgFinder, '__call__')
    def test_smartBackground_low_limit_inp_not_none(self, mock_call, mock_gaussFit):
        g = (1, 2)  # The Gauss value
        mock_gaussFit.return_value = g
        mock_call.return_value = 2

        b = self.create_finder()
        b.gauss = g
        r = b.smartBackground(inp=7)

        self.assertEqual(r, 2)
        mock_call.assert_called_once_with('fraserMode', 7)

