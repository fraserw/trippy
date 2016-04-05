# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
from unittest import TestCase
from mock import (MagicMock, patch, )
from trippy.bgFinder import bgFinder

class TestBGFinder(TestCase):

    @patch('trippy.bgFinder.num.median')
    def test_call_default(self, mock_median):
        'Test the default values of the ``__call__`` method'
        b = bgFinder(MagicMock())
        b.data = MagicMock()
        r = b()

        mock_median.assert_called_once_with(b.data)
        self.assertEqual(mock_median(), r)
