#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import unittest

from scine_art.test.resources import test_resource_path
from scine_art.io import load_file
from scine_art.molecules import (
    get_atom_mapping,
    maximum_matching_fragments,
    masm_complete_graph_match,
    intersection,
    set_difference,
    symmetric_difference
)


class Molecules(unittest.TestCase):

    def test_get_atom_mapping_same_molecule(self):
        fpath = os.path.join(test_resource_path(), 'various', 'benzene.xyz')
        molecules = load_file(fpath)
        assert len(molecules) == 1
        benzene = molecules[0]
        mapping = get_atom_mapping(benzene, benzene)
        for i, j in enumerate(mapping):
            assert i == j

    def test_maximum_matching_fragments(self):
        benzene = load_file(os.path.join(test_resource_path(), 'various', 'benzene.xyz'))[0]
        ph = load_file(os.path.join(test_resource_path(), 'various', 'ph.xyz'))[0]
        results = maximum_matching_fragments(ph, benzene)
        # Number of fragments
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert len(results[2]) == 1
        # Only one fragment per match in this case
        for frag in results[2]:
            assert len(frag) == 12
            for atom_lists in frag:
                assert len(atom_lists) == 11

    def test_masm_complete_graph_match(self):
        benzene = load_file(os.path.join(test_resource_path(), 'various', 'benzene.xyz'))[0]
        ph = load_file(os.path.join(test_resource_path(), 'various', 'ph.xyz'))[0]
        results = masm_complete_graph_match(ph, benzene)
        assert len(results) == 12

    def test_intersection(self):
        benzene = load_file(os.path.join(test_resource_path(), 'various', 'benzene.xyz'))[0]
        ph = load_file(os.path.join(test_resource_path(), 'various', 'ph.xyz'))[0]
        results = intersection(ph, benzene)
        for ref, frag in results[1]:
            assert len(ref) > 0
            if len(ref) > 1:
                for atom_lists in ref:
                    assert len(atom_lists) == len(ref[0])
            for atom_lists in frag:
                assert len(atom_lists) == len(ref[0])

    def test_set_difference(self):
        ph_oh = load_file(os.path.join(test_resource_path(), 'various', 'ph-oh.xyz'))[0]
        ph_nh2 = load_file(os.path.join(test_resource_path(), 'various', 'ph-nh2.xyz'))[0]
        results = set_difference(ph_oh, ph_nh2)
        assert len(results[0]) == 1
        assert results[0][0].graph.V == 2
        assert len(results[1]) == 1
        assert len(results[1][0]) == 1
        assert len(results[1][0][0]) == 2
        for i, j in enumerate(results[1][0][0]):
            assert results[0][0].graph.elements()[i] == ph_oh.graph.elements()[j]

    def test_symmetric_difference(self):
        ph_oh = load_file(os.path.join(test_resource_path(), 'various', 'ph-oh.xyz'))[0]
        ph_nh2 = load_file(os.path.join(test_resource_path(), 'various', 'ph-nh2.xyz'))[0]
        results = symmetric_difference(ph_oh, ph_nh2)
        assert len(results[0]) == 2
        assert results[0][0].graph.V == 2
        assert results[0][1].graph.V == 3
        assert len(results[1]) == 2
        assert len(results[1][0]) == 2
        assert len(results[1][1]) == 2
        assert len(results[1][0][0]) == 1
        assert len(results[1][0][0][0]) == 2
        for i, j in enumerate(results[1][0][0][0]):
            assert results[0][0].graph.elements()[i] == ph_oh.graph.elements()[j]
        assert results[1][0][1] == []
        assert results[1][1][0] == []
        assert len(results[1][1][1]) == 1
        assert len(results[1][1][1][0]) == 3
        for i, j in enumerate(results[1][1][1][0]):
            assert results[0][1].graph.elements()[i] == ph_nh2.graph.elements()[j]
