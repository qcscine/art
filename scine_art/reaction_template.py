#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


import scine_molassembler as masm
import scine_utilities as utils

from scine_art.molecules import (
    masm_complete_graph_match,
    sort_indices_by_subgraph,
    mol_from_subgraph_indices,
)

from dataclasses import dataclass, asdict

from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
    TypeVar,
    Set,
    Generic
)
import sys
if sys.version_info < (3, 10):
    TypeAlias = Any
else:
    from typing import TypeAlias
from copy import deepcopy, copy
from collections import Counter
from itertools import permutations, product
from networkx import Graph
from networkx.algorithms import isomorphism
from uuid import uuid4
import numpy as np

T = TypeVar('T')


@dataclass
class SideContainer(Generic[T]):
    """
    A dictionary like container with the keys ``lhs`` and ``rhs`` indicating
    the two sides of a reaction. For properties encoding mappings from one side
    to the other, the given key gives the side that if mapped from.

    Examples
    --------
    >>> counters: SideContainer[int] = ...
    >>> print(f'Number of molecules on the left-hand side: {counters["lhs"]}')
    >>> print(f'Number of molecules on the right-hand side: {counters["rhs"]}')

    """
    lhs: T
    rhs: T

    def __getitem__(self, key: str) -> T:
        return getattr(self, key)

    def __setitem__(self, key: str, value: T):
        setattr(self, key, value)
        return


@dataclass
class TemplateTypeContainer(Generic[T]):
    """
    A dictionary like container for different template sub-types.
    Depending on the requested types upon analysis the keys
    ``['minimal', 'minimal_shell', 'fragment', 'fragment_shell']``,
    may be populated with data.
    """
    minimal: Optional[SideContainer[T]] = None
    minimal_shell: Optional[SideContainer[T]] = None
    fragment: Optional[SideContainer[T]] = None
    fragment_shell: Optional[SideContainer[T]] = None

    def __getitem__(self, key: str) -> SideContainer[T]:
        value = getattr(self, key)
        if value is not None:
            return value
        else:
            raise KeyError(f'No key {key} in TemplateTypeContainer.')

    def __setitem__(self, key: str, value: SideContainer[T]):
        setattr(self, key, value)
        return


@dataclass
class AssosDissos(Generic[T]):
    """
    A dictionary like container with the keys ``assos`` and ``dissos``
    indicating data based on forming and breaking bonds.
    """
    assos: List[T]
    dissos: List[T]

    def __getitem__(self, key: str) -> List[T]:
        return getattr(self, key)

    def __setitem__(self, key: str, value: List[T]):
        setattr(self, key, value)
        return


MoFrAtIndices: TypeAlias = Tuple[int, int, int]
"""
A triplet of integers pointing at a specific atom within a template.
Templates group atoms into fragments and fragments into molecule, this triplet
encodes this grouping ``(<mol_idx>, <frag_idx>, <atom_idx>)``.
The first integer indicating the index of the molecule the atom is in,
the second index pointing to the fragment within the molecule, the third
pointing at an atom within this fragment.
"""
MoFrAtPair: TypeAlias = Tuple[MoFrAtIndices, MoFrAtIndices]
"""
Matches two atoms. Both atoms are encoded based on the molecule-fragment-atom
grouping.
"""
BondChangesContainer: TypeAlias = SideContainer[AssosDissos[MoFrAtPair]]
"""
A nested dictionary encoding all bond associations (``'assos'``) and
dissociations (``'dissos'``) that are part of a reaction template.
Bond modifications are given per side of the reaction (outer dictionary key).

Examples
--------
>>> bond_changes: BondChangesContainer = ...
>>> (mol_idx1, frag_idx1, atom_idx1), (mol_idx2, frag_idx2, atom_idx2) = \
>>>      bond_changes['lhs']['assos']

"""
SideConversionTree: TypeAlias = List[List[List[MoFrAtIndices]]]
"""
Maps one atom encoded by a molecule-fragment-atom index set onto another that is
encoded the same way.

Examples
--------
>>> mapping: SideConversionTree = ...
>>> rhs_mol_idx, rhs_frag_idx, rhs_atom_idx = \
>>>      mapping[lhs_mol_idx][lhs_frag_idx][lhs_atom_idx]

"""
ShapesByMoFrAt: TypeAlias = List[List[List[Optional[masm.shapes.Shape]]]]
"""
Lists atom shapes for atoms encoded by a molecule-fragment-atom triple

Examples
--------
>>> shapes: ShapesByMoFrAt = ...
>>> atom_shape = shapes[mol_idx][frag_idx][atom_idx]

"""
Match: TypeAlias = Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]
"""
Indices references the given lists of molecules in the generating call.

Examples
--------
>>> molecules = [masm_mol1, masm_mol2]
>>> match: Match = matching_function(molecules, ...)
>>> for association in match['assos']:
>>>     mol_idx1, atom_idx1 = association[0]
>>>     mol_idx2, atom_idx2 = association[1]
>>>     # connect the specified atoms to build product structures
>>>     #   (repeat with disconnections for dissociations)

"""


class ReactionTemplate:
    """
    A class encoding a single reaction template,
    with a basic constructor, requiring detailed data structures.

    For simpler methods of constructing a reaction template, please see
    the static methods of this class: ``from_trajectory_spline``,
    ``from_trajectory`` and ``from_aligned_structures``.

    Parameters
    ----------
    fragments : TemplateTypeContainer[List[List[masm.Molecule]]]
        The fragments on both sides of the reaction.
        Given once for each template flavor/type to be held.
    lhs_rhs_mappings : TemplateTypeContainer[SideConversionTree]
        The mapping of the atom between ``lhs`` and ``rhs`` of the
        templates.
    shapes : TemplateTypeContainer[ShapesByMoFrAt]
        The shapes of the ligand field at all atoms in given molecular
        fragments.
    barriers : Optional[Tuple[float, float]], optional
        The lowest known barriers in kJ/mol for forward.
        (``'lhs'`` -> ``'rhs'``) and backward (``'rhs'`` -> ``'lhs'``)
        reaction.
    elementary_step_id : Optional[str], optional
        The ID of the/an analyzed elementary step in a SCINE Database.
    """

    def __init__(
        self,
        fragments: TemplateTypeContainer[List[List[masm.Molecule]]],
        lhs_rhs_mappings: TemplateTypeContainer[SideConversionTree],
        shapes: TemplateTypeContainer[ShapesByMoFrAt],
        barriers: Optional[Tuple[float, float]] = None,
        elementary_step_id: Optional[str] = None
    ) -> None:

        self.__uuid = uuid4().hex
        # Members unique per template
        self.nucleus: SideContainer[List[List[masm.Molecule]]] = deepcopy(fragments['minimal'])
        self.lhs_rhs_mapping: SideContainer[SideConversionTree] = deepcopy(lhs_rhs_mappings['minimal'])
        # Expandable members
        if fragments.minimal_shell is not None:
            self.shelled_nuclei = SideContainer[List[List[List[masm.Molecule]]]](
                [deepcopy(fragments['minimal_shell']['lhs'])],
                [deepcopy(fragments['minimal_shell']['rhs'])]
            )
            assert len(self.shelled_nuclei['lhs']) == len(self.shelled_nuclei['rhs'])
        self.shapes = SideContainer[List[ShapesByMoFrAt]](
            [deepcopy(shapes['minimal']['lhs'])],
            [deepcopy(shapes['minimal']['rhs'])]
        )
        self.known_barriers: Tuple[List[float], List[float]] = ([], [])
        self.lowest_known_barriers: Optional[Tuple[float, float]] = None
        if barriers:
            self.known_barriers[0].append(barriers[0])
            self.known_barriers[1].append(barriers[1])
            self.lowest_known_barriers = barriers
        # ID, reverse?
        self.known_elementary_steps: Set[Tuple[str, bool]] = set()
        if elementary_step_id:
            self.known_elementary_steps.update(((elementary_step_id, False), ))
        assert len(self.shapes['lhs']) == len(self.shapes['rhs'])
        # Compute bond changes for minimal template version
        self.bond_changes: BondChangesContainer = {
            'lhs': {
                'assos': [],
                'dissos': []
            },
            'rhs': {
                'assos': [],
                'dissos': []
            }
        }
        for side in ['lhs', 'rhs']:
            other = 'lhs' if side == 'rhs' else 'rhs'
            for mol_idx, (t_mol, t_mol_maps) in enumerate(zip(self.nucleus[side], self.lhs_rhs_mapping[side])):
                for frag_idx, (t_frag, t_frag_maps) in enumerate(zip(t_mol, t_mol_maps)):
                    for i in range(t_frag.graph.V):
                        i_map = t_frag_maps[i]
                        # Check existing bonds
                        adjacents = [at for at in t_frag.graph.adjacents(i)]
                        for adj in adjacents:
                            adj_map = t_frag_maps[adj]
                            if i_map[0] == adj_map[0] and i_map[1] == adj_map[1]:
                                # Atom is still in same fragment on the other side
                                other_t_frag = self.nucleus[other][i_map[0]][i_map[1]]
                                other_side_adjacents_of_i = [at for at in other_t_frag.graph.adjacents(i_map[2])]
                                if adj_map[2] in other_side_adjacents_of_i:
                                    # Bond still exists, do nothing
                                    continue
                            # Bond is gone
                            if (mol_idx, frag_idx, i) > (mol_idx, frag_idx, int(adj)):
                                new = ((mol_idx, frag_idx, int(adj)), (mol_idx, frag_idx, i))
                            else:
                                new = ((mol_idx, frag_idx, i), (mol_idx, frag_idx, int(adj)))
                            if new not in self.bond_changes[side]['dissos']:
                                self.bond_changes[side]['dissos'].append(new)
                            if i_map > adj_map:
                                new = (adj_map, i_map)
                            else:
                                new = (i_map, adj_map)
                            if new not in self.bond_changes[other]['assos']:
                                self.bond_changes[other]['assos'].append(new)
        self.__networkx_graph: Optional[Graph] = None

    @staticmethod
    def from_trajectory_spline(
        spline: utils.bsplines.TrajectorySpline,
        reverse: bool = False,
        barriers: Optional[Tuple[float, float]] = None,
        elementary_step_id: Optional[str] = None,
    ) -> Optional['ReactionTemplate']:
        """Generates a reaction template from a molecular trajectory given as a
        spline interpolation.

        This function will *not* extract energy formation from the spline.

        Parameters
        ----------
        spline : scine_utilities.bsplines.TrajectorySpline
            The molecular trajectory.
        reverse : bool
            If true will reverse the spline before analysis, ``False`` by
            default.
        barriers : Optional[Tuple[float, float]], optional
            The lowest known barriers in kJ/mol for forward.
            (``'lhs'`` -> ``'rhs'``) and backward (``'rhs'`` -> ``'lhs'``)
            reaction.
        elementary_step_id : Optional[str], optional
            The ID of the/an analyzed elementary step in a SCINE Database.

        Returns
        -------
        Optional[ReactionTemplate]
            The generated reaction template.
        """
        if reverse:
            _, rhs_structure = spline.evaluate(0.0)
            _, lhs_structure = spline.evaluate(1.0)
        else:
            _, lhs_structure = spline.evaluate(0.0)
            _, rhs_structure = spline.evaluate(1.0)
        return ReactionTemplate.from_aligned_structures(
            lhs_structure,
            rhs_structure,
            barriers=barriers,
            elementary_step_id=elementary_step_id,
        )

    @staticmethod
    def from_trajectory(
        trajectory: utils.MolecularTrajectory,
        elementary_step_id: Optional[str] = None,
    ) -> Optional['ReactionTemplate']:
        """Generates a reaction template from a given molecular trajectory.

        This function will try to extract a transition state energy and reaction
        barriers from the trajectory. If this behaviour is not wanted make sure
        that the trajectory does not contain energies.

        Parameters
        ----------
        trajectory : scine_utilities.MolecularTrajectory
            The molecular trajectory.
        elementary_step_id : Optional[str], optional
            The ID of the/an analyzed elementary step in a SCINE Database.

        Returns
        -------
        Optional[ReactionTemplate]
            The generated reaction template.
        """
        elements = trajectory.elements
        lhs_structure: utils.AtomCollection = utils.AtomCollection(elements, trajectory[0])
        rhs_structure: utils.AtomCollection = utils.AtomCollection(elements, trajectory[trajectory.size()-1])
        barriers: Optional[Tuple[float, float]] = None
        energies = trajectory.get_energies()
        if energies:
            if len(energies) == trajectory.size():
                ts_idx = np.argmax(energies)
                ts_energy = energies[ts_idx]
                lhs_energy = energies[0]
                rhs_energy = energies[-1]
                barriers = ((ts_energy-lhs_energy)*2625.5, (ts_energy-rhs_energy)*2625.5)
                assert barriers[0] > 0.0e0
                assert barriers[1] > 0.0e0
        return ReactionTemplate.from_aligned_structures(
            lhs_structure,
            rhs_structure,
            barriers=barriers,
            elementary_step_id=elementary_step_id,
        )

    @staticmethod
    def from_aligned_structures(
        lhs_structure: utils.AtomCollection,
        rhs_structure: utils.AtomCollection,
        barriers: Optional[Tuple[float, float]] = None,
        elementary_step_id: Optional[str] = None,
    ) -> Optional['ReactionTemplate']:
        """Generates a reaction template from two aligned structures.

        This function expects the atoms within both given structures to mach.
        That is that the atoms have consistent indices.

        Parameters
        ----------
        lhs_structure : scine_utilities.AtomCollection
            The left-hand side molecules in a single structure.
        rhs_structure : scine_utilities.AtomCollection
            The right-hand side molecules in a single structure.
        barriers : Optional[Tuple[float, float]], optional
            The lowest known barriers in kJ/mol for forward.
            (``'lhs'`` -> ``'rhs'``) and backward (``'rhs'`` -> ``'lhs'``)
            reaction.
        elementary_step_id : Optional[str], optional
            The ID of the/an analyzed elementary step in a SCINE Database.

        Returns
        -------
        Optional[ReactionTemplate]
            The generated reaction template.
        """
        lhs_interpretation = masm.interpret.molecules(
            lhs_structure,
            utils.BondDetector.detect_bonds(lhs_structure),
            set(),
            {},
            masm.interpret.BondDiscretization.Binary,
        )
        rhs_interpretation = masm.interpret.molecules(
            rhs_structure,
            utils.BondDetector.detect_bonds(rhs_structure),
            set(),
            {},
            masm.interpret.BondDiscretization.Binary,
        )
        assert lhs_structure.size() == rhs_structure.size()
        for e1, e2 in zip(lhs_structure.elements, rhs_structure.elements):
            assert e1 == e2
        lhs = lhs_interpretation.molecules
        rhs = rhs_interpretation.molecules
        lhs_mapping: List[List[Tuple[int, int]]] = [[] for _ in lhs_interpretation.molecules]
        lhs_tmp = []
        lhs_counts = [0 for _ in lhs_interpretation.molecules]
        for i in range(lhs_structure.size()):
            mapped = lhs_interpretation.component_map.apply(i)
            lhs_tmp.append((mapped.component, mapped.atom_index))
            lhs_counts[mapped.component] += 1
        rhs_mapping: List[List[Tuple[int, int]]] = [[] for _ in rhs_interpretation.molecules]
        rhs_tmp = []
        rhs_counts = [0 for _ in rhs_interpretation.molecules]
        for i in range(rhs_structure.size()):
            mapped = rhs_interpretation.component_map.apply(i)
            rhs_tmp.append((mapped.component, mapped.atom_index))
            rhs_counts[mapped.component] += 1
        for i in range(lhs_structure.size()):
            lhs_mapping[lhs_tmp[i][0]].append(rhs_tmp[i])
            rhs_mapping[rhs_tmp[i][0]].append(lhs_tmp[i])
        template_data = ReactionTemplate._generate_reaction_template_data(
            lhs, rhs, (lhs_mapping, rhs_mapping),
            additional_template_types=['minimal_shell']
        )
        if template_data:
            return ReactionTemplate(
                *template_data,
                barriers=barriers,
                elementary_step_id=elementary_step_id,
            )
        else:
            return None

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __str__(self) -> str:
        return self.__repr__()

    def get_uuid(self) -> str:
        """
        Returns
        -------
        str
            A unique string identifying this template.
        """
        return self.__uuid

    # TODO requires masm.Molecule JSON representation generation on the fly
    # def to_json(self) -> str:
    #     return json.dumps(self.__dict__, indent=4)

    def determine_all_matches(
        self,
        molecules: List[masm.Molecule],
        energy_cutoff: float = 300,
        enforce_atom_shapes: bool = True,
        enforce_lhs: bool = False,
        enforce_rhs: bool = False,
        allowed_sides: Optional[List[str]] = None
    ) -> Optional[List[Match]]:
        """Tries to find matching fragment patterns of the reaction template in
        the given molecules.

        Parameters
        ----------
        molecules : List[masm.Molecule]
            The molecules to apply the template to.
        energy_cutoff : float, optional
            Only apply the template if a barrier in the trialed direction of
            less than this value has been reported, by default 300 (kJ/mol)
        enforce_atom_shapes : bool, optional
            If true only allow atoms with the same coordination sphere shapes
            to be considered matching, by default ``True``.
        enforce_lhs : bool, optional
            Enforce the trial of the left-hand side fragments of the reaction
            template independent of the ``energy_cutoff``, by default False
        enforce_rhs : bool, optional
            Enforce the trial of the right-hand side fragments of the reaction
            template independent of the ``energy_cutoff``, by default False
        allowed_sides : Optional[List[str]], optional
            Filter the allowed sides to be tested, by default None,
            indicating both left-hand side (``lhs``) and right-hand side
            (``rhs``) are allowed.

        Returns
        -------
        Optional[List[Match]]
            A list of matches that were found.
        """
        sides = []
        if allowed_sides is None:
            allowed_sides = ['lhs', 'rhs']
        if self.lowest_known_barriers is not None:
            if 'lhs' in allowed_sides:
                if self.lowest_known_barriers[0] < energy_cutoff or enforce_lhs:
                    sides.append('lhs')
            if 'rhs' in allowed_sides:
                if self.lowest_known_barriers[1] < energy_cutoff or enforce_rhs:
                    sides.append('rhs')
        else:
            sides = allowed_sides
        if not sides:
            return None
        # check number of molecules
        remaining_sides = []
        for side in sides:
            if len(self.nucleus[side]) == len(molecules):
                remaining_sides.append(side)
        if not remaining_sides:
            return None
        # check for atom types and count
        sides = copy(remaining_sides)
        remaining_sides = []
        atom_type_counters = []
        for mol in molecules:
            atom_type_counters.append(Counter(mol.graph.elements()))
        for side in sides:
            this_side_works = True
            for t_mol in self.nucleus[side]:
                t_mol_counter: Counter = Counter()
                for t_frag in t_mol:
                    frag_counter = Counter(t_frag.graph.elements())
                    t_mol_counter += frag_counter
                a_mol_works = False
                for mol_counter in atom_type_counters:
                    this_mol_works = True
                    for key in t_mol_counter:
                        if key not in mol_counter:
                            this_mol_works = False
                        elif mol_counter[key] < t_mol_counter[key]:
                            this_mol_works = False
                    if this_mol_works:
                        a_mol_works = True
                        break
                if not a_mol_works:
                    this_side_works = False
                    break
            if this_side_works:
                remaining_sides.append(side)
        if not remaining_sides:
            return None

        def check_shape(side, t_mol_idx, t_frag_idx, t_atom_idx, shape) -> bool:
            for shape_options in self.shapes[side]:
                t_shape = shape_options[t_mol_idx][t_frag_idx][t_atom_idx]
                if t_shape == shape:
                    return True
            return False

        possible_reactions = []
        # try to match fragments
        for side in remaining_sides:
            side_matches = []
            this_side_matches_given_molecules = True
            # Get all matches between molecules and template molecules
            # graph_matches[t_mol][mol][t_frag][frag]
            graph_matches: List[List[List[List[List[int]]]]] = []
            for t_mol_idx, t_mol in enumerate(self.nucleus[side]):
                mol_to_t_mol_matches = []
                no_molecule_matches_template_molecule = True
                for mol in molecules:
                    # Determine if all fragments in template can be found
                    #   in the given molecule.
                    # If yes, then this molecule and template molecule can
                    #   mapped to one another.
                    frag_matches = []
                    all_fragments_are_identified_in_this_molecule = True
                    for t_frag_idx, t_frag in enumerate(t_mol):
                        result = masm_complete_graph_match(t_frag, mol)
                        if enforce_atom_shapes:
                            remaining_results = []
                            for r in result:
                                for t_atom_idx, atom_idx in enumerate(r):
                                    shape: Optional[masm.shapes.Shape] = None
                                    asp = mol.stereopermutators.option(atom_idx)
                                    if asp:
                                        shape = asp.shape
                                    if not check_shape(side, t_mol_idx, t_frag_idx, t_atom_idx, shape):
                                        break
                                else:
                                    remaining_results.append(r)
                            result = copy(remaining_results)
                        if result:
                            frag_matches.append(result)
                        else:
                            all_fragments_are_identified_in_this_molecule = False
                            frag_matches = []
                            break
                    if all_fragments_are_identified_in_this_molecule:
                        no_molecule_matches_template_molecule = False
                    mol_to_t_mol_matches.append(frag_matches)
                if no_molecule_matches_template_molecule:
                    this_side_matches_given_molecules = False
                    break
                else:
                    graph_matches.append(mol_to_t_mol_matches)
            if not this_side_matches_given_molecules:
                continue
            # Given n molecules and n template molecules we check all
            #   nxn permutations that would be possible over-all-matches and
            #   remove those where molecule-to-template-molecule matches
            #   are missing
            possible_mol_permutations = list(permutations(range(len(graph_matches))))
            for p_index in reversed(range(len(possible_mol_permutations))):
                mol_permutation = possible_mol_permutations[p_index]
                for i, p in enumerate(mol_permutation):
                    if not graph_matches[i][p]:
                        possible_mol_permutations.pop(p_index)
                        break
            if not possible_mol_permutations:
                continue
            # For all remaining permutations of molecule-to-template-molecule
            #   mappings we now analyze all possible fragment combinations
            for mol_permutation in possible_mol_permutations:
                fragments_lists = []
                mol_index_lists = []
                for t_mol_idx, mol_idx in enumerate(mol_permutation):
                    for frag in graph_matches[t_mol_idx][mol_idx]:
                        assert frag is not None
                        fragments_lists.append(frag)
                        mol_index_lists.append(mol_idx)
                for atom_idxs in product(*fragments_lists):
                    # By definition of our templates, fragments
                    #   1) may not overlap
                    #   2) may not be adjacent such that they are actually
                    #      one larger fragment
                    fragment_combination_is_allowed = True
                    for mol_idx in range(max(mol_index_lists) + 1):
                        def check_fragments(idx, atom_idxs, mol_index_lists) -> bool:
                            fragments_of_this_mol = [a for a, i in zip(atom_idxs, mol_index_lists) if i == idx]
                            for f1_idx, frag1 in enumerate(fragments_of_this_mol):
                                for f2_idx in range(f1_idx):
                                    frag2 = fragments_of_this_mol[f2_idx]
                                    if (set(frag1) & set(frag2)):
                                        # Same atom in different fragments
                                        return False
                                    for atom_idx in frag1:
                                        adjacents = [at for at in molecules[idx].graph.adjacents(atom_idx)]
                                        if (set(adjacents) & set(frag2)):
                                            # Fragments are adjacent
                                            return False
                            return True
                        if not check_fragments(mol_idx, atom_idxs, mol_index_lists):
                            fragment_combination_is_allowed = False
                            break
                    if fragment_combination_is_allowed:
                        side_matches.append((side, atom_idxs, mol_index_lists))
            # match bond generation and breaking
            t_frag_to_t_mol = []
            for i, t_mol in enumerate(self.nucleus[side]):
                for t_frag in t_mol:
                    t_frag_to_t_mol.append(i)
            for match in side_matches:
                reaction: Match = {
                    'assos': [],
                    'dissos': []
                }
                for assos in self.bond_changes[match[0]]['assos']:
                    frag_idx_1 = [i for i, n in enumerate(t_frag_to_t_mol) if n == assos[0][0]][assos[0][1]]
                    frag_idx_2 = [i for i, n in enumerate(t_frag_to_t_mol) if n == assos[1][0]][assos[1][1]]
                    atom_tuple_1 = (match[2][frag_idx_1], match[1][frag_idx_1][assos[0][2]])
                    atom_tuple_2 = (match[2][frag_idx_2], match[1][frag_idx_2][assos[1][2]])
                    if atom_tuple_1 > atom_tuple_2:
                        reaction['assos'].append((atom_tuple_2, atom_tuple_1))
                    else:
                        reaction['assos'].append((atom_tuple_1, atom_tuple_2))
                for dissos in self.bond_changes[match[0]]['dissos']:
                    frag_idx_1 = [i for i, n in enumerate(t_frag_to_t_mol) if n == dissos[0][0]][dissos[0][1]]
                    frag_idx_2 = [i for i, n in enumerate(t_frag_to_t_mol) if n == dissos[1][0]][dissos[1][1]]
                    atom_tuple_1 = (match[2][frag_idx_1], match[1][frag_idx_1][dissos[0][2]])
                    atom_tuple_2 = (match[2][frag_idx_2], match[1][frag_idx_2][dissos[1][2]])
                    if atom_tuple_1 > atom_tuple_2:
                        reaction['dissos'].append((atom_tuple_2, atom_tuple_1))
                    else:
                        reaction['dissos'].append((atom_tuple_1, atom_tuple_2))
                possible_reactions.append(reaction)
        if not possible_reactions:
            return None
        else:
            return possible_reactions

    def apply(
        self,
        molecules: List[masm.Molecule],
        energy_cutoff: float = 300,
        enforce_atom_shapes: bool = True,
        enforce_forward: bool = False,
        enforce_backward: bool = False,
        allowed_directions: Optional[List[str]] = None
    ) -> List[
        Tuple[List[masm.Molecule],
              List[List[Tuple[int, int]]],
              masm.Molecule,
              List[List[Tuple[int, int]]],
              Match]]:
        """Tries to apply a given template to the presented molecules.

        Parameters
        ----------
        molecules : List[masm.Molecule]
            The molecules to apply the template to.
        energy_cutoff : float, optional
            Only apply the template if a barrier in the trialed direction of
            less than this value has been reported, by default 300 (kJ/mol)
        enforce_atom_shapes : bool, optional
            If true only allow atoms with the same coordination sphere shapes
            to be considered matching, by default ``True``.
        enforce_forward : bool, optional
            Enforce the trial of the forward direction independent of the
            ``energy_cutoff``, by default False
        enforce_backward : bool, optional
            Enforce the trial of the backward direction independent of the
            ``energy_cutoff``, by default False
        allowed_directions : List[str], optional
            Filter the allowed directions to be trial, by default None,
            indicating both ``forward`` and ``backward`` directions are allowed.

        Returns
        -------
        List[Tuple[List[masm.Molecule], List[List[Tuple[int, int]]], masm.Molecule, List[List[Tuple[int, int]]], Match]]
            Results of all possible applications of the given template, one
            tuple for each possibility.
            Each tuple contains:
            0. The list of resulting molecules
            1. The mapping of all atoms from given molecules to those in the
            resulting products (0.).
            2. The fused transition state, where all associative bond
            modifications are applied but non of the dissociate ones.
            3. The mapping of all atoms from given molecules to those in the
            fused transition state (2.).
        """
        allowed_sides = []
        if allowed_directions is None:
            allowed_directions = ['forward', 'backward']
        if 'forward' in allowed_directions:
            allowed_sides.append('lhs')
        if 'backward' in allowed_directions:
            allowed_sides.append('rhs')
        if not allowed_directions:
            return []
        matches = self.determine_all_matches(
            molecules=molecules,
            energy_cutoff=energy_cutoff,
            enforce_atom_shapes=enforce_atom_shapes,
            enforce_lhs=enforce_forward,
            enforce_rhs=enforce_backward,
            allowed_sides=allowed_sides
        )
        if matches is None:
            return []
        results = []
        for match in matches:
            resulting_mols = [copy(mol) for mol in molecules]
            current_mapping = []
            for i, mol in enumerate(molecules):
                current_mapping.append([(i, j) for j in range(mol.graph.V)])
            for (mol1, atom1), (mol2, atom2) in match['assos']:
                mapped_mol1_idx, mapped_atom1_idx = current_mapping[mol1][atom1]
                mapped_mol2_idx, mapped_atom2_idx = current_mapping[mol2][atom2]
                assert mapped_mol1_idx <= mapped_mol2_idx
                if mapped_mol1_idx != mapped_mol2_idx:
                    # The new bond will fuse two fragments
                    new_molecule = masm.editing.add_ligand(
                        resulting_mols[mapped_mol1_idx],
                        resulting_mols[mapped_mol2_idx],
                        mapped_atom1_idx,
                        [mapped_atom2_idx]
                    )
                    # Rebuild current mapping
                    offset = resulting_mols[mapped_mol1_idx].graph.V
                    current_mapping[mol2] = [(mapped_mol1_idx, i+offset) for (_, i) in current_mapping[mol2]]
                    for k, mol_map in enumerate(current_mapping):
                        if mol_map[0][0] > mapped_mol2_idx:
                            current_mapping[k] = [(i-1, j) for (i, j) in mol_map]
                    # Update resulting molecules
                    resulting_mols[mapped_mol1_idx] = new_molecule
                    resulting_mols.pop(mapped_mol2_idx)
                else:
                    resulting_mols[mapped_mol1_idx].add_bond(mapped_atom1_idx, mapped_atom2_idx)
                assert len(resulting_mols) == 1
            ts = deepcopy(resulting_mols[0])
            ts_atom_mapping = deepcopy(current_mapping)
            for (mol1, atom1), (mol2, atom2) in match['dissos']:
                mapped_mol1_idx, mapped_atom1_idx = current_mapping[mol1][atom1]
                mapped_mol2_idx, mapped_atom2_idx = current_mapping[mol2][atom2]
                assert mapped_mol1_idx == mapped_mol2_idx
                assert mapped_atom1_idx != mapped_atom2_idx
                if resulting_mols[mapped_mol1_idx].graph.can_remove(masm.BondIndex(mapped_atom1_idx, mapped_atom2_idx)):
                    # Simply remove the bond
                    resulting_mols[mapped_mol1_idx].remove_bond(mapped_atom1_idx, mapped_atom2_idx)
                else:
                    # The bond removal will cleave the molecules in two,
                    #  track the fragments
                    cleaved = masm.editing.cleave(
                        resulting_mols[mapped_mol1_idx],
                        masm.BondIndex(mapped_atom1_idx, mapped_atom2_idx)
                    )
                    # Rebuild current mapping
                    for i, mol_map in enumerate(current_mapping):
                        for j, (mapped_mol_idx, mapped_atom_idx) in enumerate(mol_map):
                            if mapped_mol_idx == mapped_mol1_idx:
                                # The component map lists the new molecules as {0,1}
                                #   transform to the new molecule indices:
                                #   {<current index>, <new element in list>}
                                if cleaved.component_map[mapped_atom_idx][0] == 0:
                                    mol_map[j] = (mapped_mol1_idx, cleaved.component_map[mapped_atom_idx][1])
                                else:
                                    mol_map[j] = (len(resulting_mols), cleaved.component_map[mapped_atom_idx][1])
                        current_mapping[i] = mol_map
                    # Update resulting molecules
                    resulting_mols[mapped_mol1_idx] = cleaved.first
                    resulting_mols.append(cleaved.second)
            results.append((
                resulting_mols,
                current_mapping,
                ts,
                ts_atom_mapping,
                match
            ))
        return results

    def get_networkx_graph(self) -> Tuple[Graph, dict]:
        """Get the graph representation of this reaction template.

        Returns
        -------
        Tuple[networkx.Graph, dict]
            Contains the ``networkx.Graph`` and a mapping between graph nodes
            atoms in the template.
        """
        if self.__networkx_graph is None:
            self.__networkx_graph = self.generate_networkx_graph_from_template(self)
        return self.__networkx_graph

    @staticmethod
    def __node_match(n1, n2) -> bool:
        if n1['type'] != n2['type']:
            return False
        if n1['type'] == 'atom':
            return bool(n1['at'] == n2['at'])
        else:
            return True

    @staticmethod
    def __edge_match(e1, e2) -> bool:
        return bool(e1['weight'] == e2['weight'])

    @staticmethod
    def generate_networkx_graph_from_template(template: 'ReactionTemplate') -> Tuple[Graph, dict]:
        """Generates a graph representation of the reaction template.

        Parameters
        ----------
        template : ReactionTemplate
            The template to be converted.

        Returns
        -------
        Tuple[networkx.Graph, dict]
            Contains the ``networkx.Graph`` and a mapping between graph nodes
            atoms in the template.
        """
        graph = Graph()
        # Add reaction arrow nodes
        graph.add_nodes_from([0, 1], type='rxn')
        graph.add_edge(0, 1, weight=1)
        node_indices = SideContainer[Dict[str, Any]](
            {
                'lhs': 0,
                'mol': [],
                'frag': [],
                'atoms': []
            },
            {
                'rhs': 1,
                'mol': [],
                'frag': [],
                'atoms': []
            }
        )
        counter = 1
        for side in ['lhs', 'rhs']:
            for m, mol in enumerate(template.nucleus[side]):
                # Add molecule nodes
                counter += 1
                mol_counter = counter
                graph.add_nodes_from([counter], type='mol')
                graph.add_edge(node_indices[side][side], counter, weight=2)
                node_indices[side]['mol'].append(mol_counter)
                fragment_counter = []
                outer_atom_counters = []
                for f, frag in enumerate(mol):
                    # Add fragment nodes
                    counter += 1
                    frag_counter = counter
                    graph.add_nodes_from([counter], type='frag')
                    graph.add_edge(mol_counter, counter, weight=3)
                    fragment_counter.append(frag_counter)
                    atom_counters = []
                    for i, element in enumerate(frag.graph.elements()):
                        # Add atom nodes
                        counter += 1
                        # TODO Atom type
                        graph.add_nodes_from([counter], type='atom', at=element)
                        graph.add_edge(frag_counter, counter, weight=4)
                        atom_counters.append(counter)
                    for i, element in enumerate(frag.graph.elements()):
                        adjacents = [at for at in frag.graph.adjacents(i)]
                        for a in adjacents:
                            # Add constant bonds and dissociations
                            if a < i and a in atom_counters:
                                if ((m, f, a), (m, f, i)) in template.bond_changes[side]['dissos']:
                                    graph.add_edge(atom_counters[a], atom_counters[i], weight=6)
                                elif ((m, f, i), (m, f, a)) in template.bond_changes[side]['dissos']:
                                    graph.add_edge(atom_counters[a], atom_counters[i], weight=6)
                                else:
                                    graph.add_edge(atom_counters[a], atom_counters[i], weight=5)
                    outer_atom_counters.append(atom_counters)
                node_indices[side]['frag'].append(fragment_counter)
                node_indices[side]['atoms'].append(outer_atom_counters)
            for asso in template.bond_changes[side]['assos']:
                graph.add_edge(
                    node_indices[side]['atoms'][asso[0][0]][asso[0][1]][asso[0][2]],
                    node_indices[side]['atoms'][asso[1][0]][asso[1][1]][asso[1][2]],
                    weight=7
                )
        for m, mol in enumerate(template.nucleus['lhs']):
            for f, frag in enumerate(mol):
                for a, _ in enumerate(frag.graph.elements()):
                    tup = template.lhs_rhs_mapping['lhs'][m][f][a]
                    graph.add_edge(
                        node_indices['lhs']['atoms'][m][f][a],
                        node_indices['rhs']['atoms'][tup[0]][tup[1]][tup[2]],
                        weight=8
                    )
        return graph, asdict(node_indices)

    def __eq__(self, other) -> bool:
        """Compares the two template graphs.

        Parameters
        ----------
        other : ReactionTemplate
            The other reaction template to be checked.

        Returns
        -------
        bool
            ``True`` if the two templates are identical.
        """
        if not isinstance(other, ReactionTemplate):
            return NotImplemented
        g1, _ = self.get_networkx_graph()
        g2, _ = other.get_networkx_graph()
        gm = isomorphism.GraphMatcher(g1, g2, node_match=self.__node_match, edge_match=self.__edge_match)
        return gm.is_isomorphic()

    def strict_equal(self, other: 'ReactionTemplate') -> bool:
        """At the moment this function is identical to the basic ``__eq__``
        operator, in future versions it may contain additional checks.

        Parameters
        ----------
        other : ReactionTemplate
            The other reaction template to be checked.

        Returns
        -------
        bool
            ``True`` if the two templates are identical.
        """
        if self != other:
            return False
        # TODO check shapes and more
        return True

    def update(self, other: 'ReactionTemplate') -> None:
        """If the two templates are identical (isomorphic ``minimal`` template
        graphs), the data of the argument will be subsumed into the base
        template.

        This means shape information, recorded energy barriers and known
        elementary steps will be combined.

        Parameters
        ----------
        other : ReactionTemplate
            The other reaction template to update this with.
        """
        if not isinstance(other, ReactionTemplate):
            return NotImplemented
        g1, g1_mapping = self.get_networkx_graph()
        g2, g2_mapping = other.get_networkx_graph()
        gm = isomorphism.GraphMatcher(g1, g2, node_match=self.__node_match, edge_match=self.__edge_match)
        matches_existing = gm.is_isomorphic()
        if matches_existing:
            matches_reverse = bool(gm.mapping[0] == 1)
            if other.known_barriers:
                if matches_reverse:
                    self.known_barriers = (
                        self.known_barriers[0] + other.known_barriers[0],
                        self.known_barriers[1] + other.known_barriers[1]
                    )
                else:
                    self.known_barriers = (
                        self.known_barriers[0] + other.known_barriers[0],
                        self.known_barriers[1] + other.known_barriers[1]
                    )
                self.lowest_known_barriers = (
                    min(self.known_barriers[0]),
                    min(self.known_barriers[1])
                )
            if other.known_elementary_steps:
                self.known_elementary_steps.update(other.known_elementary_steps)
            shapes = self.__get_resorted_shapes(
                other,
                g1_mapping, g2_mapping,
                gm.mapping
            )
            for side in ['lhs', 'rhs']:
                self.shapes[side].extend(shapes[side])

    def __get_resorted_shapes(
        self, other, g1_mapping, g2_mapping, g1_g2_mapping
    ) -> SideContainer[List[ShapesByMoFrAt]]:
        shapes = SideContainer[List[ShapesByMoFrAt]]([], [])

        def nested_search(nested_lists, value) -> MoFrAtIndices:
            for sub_list in nested_lists:
                for sub_sub_list in sub_list:
                    if value in sub_sub_list:
                        return (
                            nested_lists.index(sub_list),
                            sub_list.index(sub_sub_list),
                            sub_sub_list.index(value)
                        )
            assert False

        # TODO optimization: only generate the mapping once, then apply it
        #                    many times over.
        for side in ['lhs', 'rhs']:
            other_side = side
            if g1_g2_mapping[0] == 1:
                other_side = 'lhs' if side == 'rhs' else 'rhs'
            for other_shapes in other.shapes[other_side]:
                mol_shapes = []
                for m, mol in enumerate(self.nucleus[side]):
                    frag_shapes = []
                    for f, frag in enumerate(mol):
                        atom_shapes = []
                        for a, _ in enumerate(frag.graph.elements()):
                            # Resolve mapping:
                            #   self -> self-graph -> other-graph -> other
                            #   m,f,a -[g1_mapping]-> x -[g1_g2_mapping]-> y -[g2_mapping]-> om,of,oa
                            x = g1_mapping[side]['atoms'][m][f][a]
                            y = g1_g2_mapping[x]
                            om, of, oa = nested_search(g2_mapping[other_side]['atoms'], y)
                            atom_shapes.append(other_shapes[om][of][oa])
                        assert len(atom_shapes) == len(self.shapes[side][0][m][f])
                        frag_shapes.append(atom_shapes)
                    if len(mol) == 0:
                        frag_shapes = [[]]
                    assert len(frag_shapes) == len(self.shapes[side][0][m])
                    mol_shapes.append(frag_shapes)
                assert len(mol_shapes) == len(self.shapes[side][0])
                if (mol_shapes not in shapes[side]) and (mol_shapes not in self.shapes[side]):
                    shapes[side].append(mol_shapes)
        return shapes

    @staticmethod
    def _generate_reaction_template_data(
        lhs: List[masm.Molecule],
        rhs: List[masm.Molecule],
        mapping: Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]],
        additional_template_types: Optional[List[str]] = None
    ) -> Optional[Tuple[
        TemplateTypeContainer[List[List[masm.Molecule]]],
        TemplateTypeContainer[SideConversionTree],
        TemplateTypeContainer[ShapesByMoFrAt],
    ]]:
        """Generate the basic data required for the constructor based on more

        Parameters
        ----------
        lhs : List[scine_molassembler.Molecule]
            The molecules on the left-hand side of the reaction.
        rhs : List[scine_molassembler.Molecule]
            The molecules on the right-hand side of the reaction.
        mapping : Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]
            The mappings of atoms between the right and left-hand side of the
            reaction. First list of lists mapping from left to right, the second
            mapping the other way.
        additional_template_types : Optional[List[str]], optional
            A list of other additional template flavors to be extracted.
            Possible values are any combination of:
            ``['minimal_shell', 'fragment', 'fragment_shell']``

        Returns
        -------
        Optional[Tuple[ TemplateTypeContainer[List[List[masm.Molecule]]],
                TemplateTypeContainer[SideConversionTree],
                TemplateTypeContainer[ShapesByMoFrAt], ]]
            The data required to build a reaction template.

        Raises
        ------
        RuntimeError
            If the template types requested are not available.
        """
        template_types = ['minimal']
        if additional_template_types is not None:
            template_types = ['minimal'] + additional_template_types
            for key in additional_template_types:
                if key not in ['minimal_shell', 'fragment', 'fragment_shell']:
                    raise RuntimeError(
                        "Only ['minimal_shell', 'fragment', 'fragment_shell'] are allowed template_types."
                    )
        if ('fragment_shell' in template_types) and ('fragment' not in template_types):
            template_types.append('fragment')

        molecules = {
            'lhs': lhs,
            'rhs': rhs,
        }
        mappings = {
            'lhs': mapping[0],
            'rhs': mapping[1],
        }
        # Tally all atoms that are connected differently on the other side

        def same_environment(lhs_atom: Tuple[int, int], rhs_atom: Tuple[int, int]) -> bool:
            lhs_mol = lhs[lhs_atom[0]]
            rhs_mol = rhs[rhs_atom[0]]
            l: int = lhs_atom[1]
            r: int = rhs_atom[1]
            assert lhs_mol.graph.element_type(l) == rhs_mol.graph.element_type(r)
            l_adjacents = [a for a in lhs_mol.graph.adjacents(l)]
            r_adjacents = [a for a in rhs_mol.graph.adjacents(r)]
            if len(l_adjacents) != len(r_adjacents):
                return False
            for a in l_adjacents:
                if mapping[0][lhs_atom[0]][a][0] != rhs_atom[0]:
                    return False
                if mapping[0][lhs_atom[0]][a][1] not in r_adjacents:
                    return False
            return True

        indices = TemplateTypeContainer[List[Set[int]]]()
        for key in template_types:
            indices[key] = SideContainer([], [])

        for i, l_mol in enumerate(lhs):
            tmp: Set[int] = set()
            for j in range(l_mol.graph.V):
                if not same_environment((i, j), mapping[0][i][j]):
                    tmp.add(j)
            indices['minimal']['lhs'].append(tmp)
        if sum(len(x) for x in indices['minimal']['lhs']) == 0:
            return None
        for i, r_mol in enumerate(rhs):
            tmp = set()
            for j in range(r_mol.graph.V):
                if not same_environment(mapping[1][i][j], (i, j)):
                    tmp.add(j)
                    assert mapping[1][i][j][1] in indices['minimal']['lhs'][mapping[1][i][j][0]]
            indices['minimal']['rhs'].append(tmp)
        if sum(len(x) for x in indices['minimal']['rhs']) == 0:
            return None

        if 'minimal_shell' in template_types:
            # Add first shell of adjacents to minimal index sets
            indices['minimal_shell']['lhs'] = deepcopy(indices['minimal']['lhs'])
            indices['minimal_shell']['rhs'] = deepcopy(indices['minimal']['rhs'])
            for l_idx, idxs in enumerate(indices['minimal_shell']['lhs']):
                first_shell: Set[int] = set()
                for atom in idxs:
                    first_shell.update(set([a for a in lhs[l_idx].graph.adjacents(atom)]))
                idxs.update(first_shell)
            for r_idx, idxs in enumerate(indices['minimal_shell']['rhs']):
                first_shell = set()
                for atom in idxs:
                    first_shell.update(set([a for a in rhs[r_idx].graph.adjacents(atom)]))
                idxs.update(first_shell)

        # Generate fragment sorts of first two sets to determine if connections
        #  are required
        sorted_indices = TemplateTypeContainer[List[List[List[int]]]]()
        for key in template_types:
            sorted_indices[key] = SideContainer([], [])

        for mol, idxs in zip(lhs, indices['minimal']['lhs']):
            sorted_indices['minimal']['lhs'].append(sort_indices_by_subgraph(list(idxs), mol.graph))
        for mol, idxs in zip(rhs, indices['minimal']['rhs']):
            sorted_indices['minimal']['rhs'].append(sort_indices_by_subgraph(list(idxs), mol.graph))
        if 'minimal_shell' in template_types:
            for mol, idxs in zip(lhs, indices['minimal_shell']['lhs']):
                sorted_indices['minimal_shell']['lhs'].append(sort_indices_by_subgraph(list(idxs), mol.graph))
            for mol, idxs in zip(rhs, indices['minimal_shell']['rhs']):
                sorted_indices['minimal_shell']['lhs'].append(sort_indices_by_subgraph(list(idxs), mol.graph))

        if 'fragment' in template_types:
            # Generate minimal connecting fragment index sets
            #  1. Generate minimal fragment for each molecule on lhs and rhs
            #  2. If more atoms are now encompassed by one side, also add them on the other
            for side in ['lhs', 'rhs']:
                indices['fragment'][side] = deepcopy(indices['minimal'][side])
            for side in ['lhs', 'rhs']:
                other = 'lhs' if side == 'rhs' else 'rhs'
                for i_mol, (mol, idxs) in enumerate(zip(molecules[side], indices['fragment'][side])):
                    fragment_idxs = sorted_indices['minimal'][side][i_mol]
                    if len(fragment_idxs) <= 1:
                        continue
                    minimal_fragment = ReactionTemplate._minimal_single_fragment(fragment_idxs, mol.graph)
                    idxs.update(minimal_fragment)
                for i_frag, frag in enumerate(indices['fragment'][side]):
                    for atom in frag:
                        atom_idx = mappings[side][i_frag][atom][1]
                        if atom_idx not in indices['fragment'][other][mappings[side][i_frag][atom][0]]:
                            indices['fragment'][other][mappings[side][i_frag]
                                                       [atom][0]].add(mappings[side][i_frag][atom][1])

            # Add first shell of adjacents to minimal fragment index sets
            if 'fragment_shell' in template_types:
                for side in ['lhs', 'rhs']:
                    indices['fragment_shell'][side] = deepcopy(indices['fragment'][side])
                    for i_mol, idxs in enumerate(indices['fragment_shell'][side]):
                        first_shell = set()
                        for atom in idxs:
                            first_shell.update(set([a for a in molecules[side][i_mol].graph.adjacents(atom)]))
                        idxs.update(first_shell)

        templates = TemplateTypeContainer[List[List[masm.Molecule]]]()
        templates_mappings = TemplateTypeContainer[List[List[List[int]]]]()
        for key in template_types:
            templates[key] = SideContainer([], [])
            templates_mappings[key] = SideContainer([], [])

        for key in template_types:
            for side in ['lhs', 'rhs']:
                for i_frag, (mol, idxs) in enumerate(zip(molecules[side], indices[key][side])):
                    sorted_idxs = sort_indices_by_subgraph(list(idxs), mol.graph)
                    fragment_list: List[masm.Molecule] = []
                    original_indices_list: List[List[int]] = []
                    for subgraph_indices in sorted_idxs:
                        try:
                            fragment, original_indices = mol_from_subgraph_indices(subgraph_indices, mol)
                        except RuntimeError:
                            return None
                        if fragment:
                            fragment_list.append(fragment)
                            original_indices_list.append(original_indices)
                    templates[key][side].append(fragment_list)
                    templates_mappings[key][side].append(original_indices_list)

        lhs_rhs_mappings = TemplateTypeContainer[SideConversionTree]()
        for key in template_types:
            lhs_rhs_mappings[key] = SideContainer([], [])

        for key in template_types:
            for side in ['lhs', 'rhs']:
                other = 'lhs' if side == 'rhs' else 'rhs'
                template_side = templates_mappings[key][side]
                mapping_to_build = lhs_rhs_mappings[key][side]
                for i_mol in range(len(template_side)):
                    fragments = template_side[i_mol]
                    mapping_to_build.append([])
                    for i_frag in range(len(fragments)):
                        atoms = fragments[i_frag]
                        mapping_to_build[i_mol].append([])
                        for atom in range(len(atoms)):
                            other_mol, other_atom = mappings[side][i_mol][atoms[atom]]
                            for x, other_fragment in enumerate(templates_mappings[key][other][other_mol]):
                                if other_atom in other_fragment:
                                    y = other_fragment.index(other_atom)
                                    mapping_to_build[i_mol][i_frag].append((other_mol, x, y))
                                    break
                            else:
                                raise RuntimeError("Bug: Failed to match atoms in template.")

        # shapes[key][side][mol][frag][atom] -> shape of atom
        shapes = TemplateTypeContainer[ShapesByMoFrAt]()
        for key in template_types:
            shapes[key] = SideContainer([], [])

        # Save shape information
        for key in template_types:
            for side in ['lhs', 'rhs']:
                for m, mol_idxs in enumerate(templates_mappings[key][side]):
                    shapes[key][side].append([])
                    for frag_idxs in mol_idxs:
                        shapes[key][side][-1].append([])
                        for atom in frag_idxs:
                            asp = molecules[side][m].stereopermutators.option(atom)
                            if asp:
                                shapes[key][side][-1][-1].append(asp.shape)
                            else:
                                shapes[key][side][-1][-1].append(None)
                        assert len(shapes[key][side][-1][-1]) > 0
                    # mol_idxs can be empty for molecules that are present in reaction
                    #   but do not actively participate
                    # TODO drop these "extra" molecules from the template?
                    if not mol_idxs:
                        shapes[key][side][-1].append([])
                    assert len(shapes[key][side][-1]) > 0

        return templates, lhs_rhs_mappings, shapes

    @staticmethod
    def _minimal_single_fragment(all_fragments: List[List[int]], graph: masm.Graph) -> Set[int]:
        for f in all_fragments:
            assert f
        fragments = deepcopy(all_fragments)
        if len(fragments) == 0:
            return set()
        elif len(fragments) == 1:
            return set(fragments[0])
        predecessors = []
        for frag in fragments:
            tmp = []
            for idx in frag:
                tmp.append(masm.shortest_paths(idx, graph))
            predecessors.append(tmp)

        # TODO Better algorithm here, this one may easily give suboptimal results
        #      for edge cases.

        # Initially choose the shortest overall path
        best_fragments: Tuple[int, int] = (0, 1)
        best_path: List[int] = predecessors[0][0].path(fragments[1][0])
        for i_frag, preds in enumerate(predecessors):
            for pred in preds:
                for j_frag, frag in enumerate(fragments):
                    if i_frag <= j_frag:
                        continue
                    for idx in frag:
                        new = pred.path(idx)
                        if len(best_path) > len(new):
                            best_path = new
                            best_fragments = (i_frag, j_frag)
        # Tally the generated total fragment so far
        connected_vertices: Set[int] = set(best_path)
        connected_vertices.update(fragments[best_fragments[0]])
        connected_vertices.update(fragments[best_fragments[1]])
        if best_fragments[1] > best_fragments[0]:
            fragments.pop(best_fragments[1])
            fragments.pop(best_fragments[0])
            predecessors.pop(best_fragments[1])
            predecessors.pop(best_fragments[0])
        else:
            fragments.pop(best_fragments[0])
            fragments.pop(best_fragments[1])
            predecessors.pop(best_fragments[0])
            predecessors.pop(best_fragments[1])
        # Continue connecting the shortest path fragments until all are connected
        while fragments:
            best_fragment: int = 0
            best_path = predecessors[0][0].path(list(connected_vertices)[0])
            for i_frag, preds in enumerate(predecessors):
                for pred in preds:
                    for idx in connected_vertices:
                        new = pred.path(idx)
                        if len(best_path) > len(new):
                            best_path = new
                            best_fragment = i_frag
            connected_vertices.update(fragments[best_fragment])
            fragments.pop(best_fragment)
            predecessors.pop(best_fragment)
        return connected_vertices
