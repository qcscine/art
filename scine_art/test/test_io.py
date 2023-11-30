#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import unittest

from scine_art.test import skip_without_dot
from scine_art.test.resources import test_resource_path
from scine_art.io import load_file, load_spline_from_trajectory, write_molecule_to_svg


class IO(unittest.TestCase):

    def test_load_file_xyz(self):
        fpath = os.path.join(test_resource_path(), 'various', 'benzene.xyz')
        molecules = load_file(fpath)
        assert len(molecules) == 1
        benzene = molecules[0]
        assert benzene.graph.V == 12
        assert benzene.graph.E == 12

    def test_spline_read(self):
        fpath = os.path.join(test_resource_path(), 'dissociation.trj.xyz')
        spline = load_spline_from_trajectory(fpath)
        assert spline.ts_position == -1.0
        assert len(spline.knots) == len(spline.data)
        _ = spline.evaluate(0.0, 3)
        _ = spline.evaluate(0.5, 3)
        _ = spline.evaluate(1.0, 3)

    @skip_without_dot()
    def test_write_molecule_to_svg(self):
        fname = 'unittest_test_write_molecule_to_svg.svg'
        if os.path.isfile(fname):
            os.remove(fname)
        assert not os.path.isfile(fname)
        fpath = os.path.join(test_resource_path(), 'various', 'benzene.xyz')
        benzene = load_file(fpath)[0]
        write_molecule_to_svg(fname, benzene)
        assert os.path.isfile(fname)
        os.remove(fname)
