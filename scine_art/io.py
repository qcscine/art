#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import scine_molassembler as masm
import scine_utilities as utils
from typing import List


def load_file(path: str) -> List[masm.Molecule]:
    """Load an atom coordinate file as ``scine_molassembler.Molecule`` s.

    If the fle format does not provide bond information, then the
    bonds will be inferred using distance criteria based on covalent radii.
    For more information on there bond inference see the
    ``scine_utilities.BondDetector``.

    Parameters
    ----------
    path : str
        Path to the structure file.

    Returns
    -------
    List[scine_molassembler.Molecule]
        The resulting interpreted molecules.

    Note
    ----
        Molecules are not canonicalized.

    """
    atoms, bonds = utils.io.read(path)
    if not bonds or bonds.empty():
        bonds = utils.BondDetector.detect_bonds(atoms)
    return masm.interpret.molecules(
        atoms,
        bonds,
        set(),
        {},
        masm.interpret.BondDiscretization.Binary,
    ).molecules


def load_spline_from_trajectory(path: str) -> utils.bsplines.TrajectorySpline:
    """Read a trajectory from an .xyz file.

    Expects energies to be given as plain floating point numbers in the
    title field of each structure

    Parameters
    ----------
    path : str
        Path to the trajectory file (.xyz)

    Returns
    -------
    utils.bsplines.TrajectorySpline
        The spline interpolation of the read trajectory.
    """
    rpi = utils.bsplines.ReactionProfileInterpolation()
    trajectory = utils.io.read_trajectory(utils.io.TrajectoryFormat.Xyz, path)
    energies = []
    with open(path, "r") as f:
        lines = f.readlines()
        nAtoms = int(lines[0].strip())
        i = 0
        while i < len(lines):
            energies.append(float(lines[i + 1].strip()))
            i += nAtoms + 2
    for pos, e in zip(trajectory, reversed(energies)):
        rpi.append_structure(utils.AtomCollection(trajectory.elements, pos), e)
    return rpi.spline(len(energies), 3)


def write_molecule_to_svg(path: str, mol: masm.Molecule) -> None:
    """Dump a single molecular graph to disk in SVG format.

    Parameters
    ----------
    path : str
        Path to the SVG file to be generated.
    mol : scine_molassembler.Molecule
        The molecule to be dumped to disk as an SVG file.
    """
    masm.io.write(path, mol)
