#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import scine_molassembler as masm

from scine_art.molecules import (
    all_matching_fragments,
    maximum_matching_fragments,
    sort_indices_by_subgraph,
    mol_from_subgraph_indices,
)

from typing import List, Tuple, Set, Optional
from copy import copy


def map_reaction_from_molecules_direct(
    lhs: List[masm.Molecule],
    rhs: List[masm.Molecule],
    known_atom_mappings: Optional[List[Tuple[int, int]]] = None
) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
    """Tries to match each atom in the molecules of the left-hand side to
    exactly one atom on the right-hand side.

    Atom type counts must match.

    Note
    ----

    The algorithm is experimental, be cautious with its results.
    This version of the algorithm is CPU time greedy, but saves memory.

    Parameters
    ----------
    lhs : List[masm.Molecule]
        A list of molecules on the left-hand side of the matching.
    rhs : List[masm.Molecule]
        A list of molecules on the right-hand side of the matching.
    known_atom_mappings : Optional[List[Tuple[int, int]]]
        A list of atom mappings that have to be respected. The atoms are
        indexed on a continuous scale in order of the atoms in the molecules
        given.

    Returns
    -------
    Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]
        A tuple atom matches. The first entry matching from left- to right-hand
        side the second from right- to left-hand side.

    Raises
    ------
    RuntimeError
        If algorithm fails or miss matches appear.
    """
    n_atoms = sum([lmol.graph.V for lmol in lhs])
    assert sum([rmol.graph.V for rmol in rhs]) == n_atoms
    # In this algorithm we will try to rebuild the LHS from fragments of
    #  the rhs. To this end the LHS will be viewed as having a continuous
    #  index with breaks where a molecule ends and a new one starts.
    #  (e.g. the input methane, HCl and benzene, in that order, would
    #  correspond to [0,1,2,3,4 | 5,6 | 7,8,9,10,11,...,18]
    # When generating the intersection based fragment mapping we will generate
    #  the mapping to this new 'continuous-LHS' index.
    l_sizes = [m.graph.V for m in lhs]
    r_sizes = [m.graph.V for m in rhs]
    l_offsets: List[int] = [sum(l_sizes[:i]) for i in range(len(lhs))]
    r_offsets: List[int] = [sum(r_sizes[:i]) for i in range(len(rhs))]

    remaining_lhs_fragments = [mol for mol in lhs]
    remaining_lhs_indices = [[x + l_offsets[i] for x in range(mol.graph.V)] for i, mol in enumerate(lhs)]
    remaining_rhs_fragments = [mol for mol in rhs]
    remaining_rhs_indices = [[x + r_offsets[i] for x in range(mol.graph.V)] for i, mol in enumerate(rhs)]

    def get_mappings():
        mappings = []
        for i, (l_mol, l_idxs) in enumerate(zip(remaining_lhs_fragments, remaining_lhs_indices)):
            for j, (r_mol, r_idxs) in enumerate(zip(remaining_rhs_fragments, remaining_rhs_indices)):
                fragments, fragments_to_l_mol, fragments_to_r_mol = maximum_matching_fragments(
                    l_mol, r_mol, min_fragment_size=1
                )
                # Generate continuous-LHS mapping, and sets of the mapped indices
                for fragment, l_maps, r_maps in zip(fragments, fragments_to_l_mol, fragments_to_r_mol):
                    for l_map in l_maps:
                        assert len(l_map) == fragment.graph.V
                        for r_map in r_maps:
                            assert len(r_map) == fragment.graph.V
                            mapping = [(l_idxs[l_map[x]], r_idxs[r_map[x]]) for x in range(fragment.graph.V)]
                            l_set = set([x[0] for x in mapping])
                            r_set = set([x[1] for x in mapping])
                            mappings.append((
                                mapping,
                                l_set,
                                r_set,
                                fragment,
                                i,
                                j,
                                [l_map[x] for x in range(fragment.graph.V)],
                                [r_map[x] for x in range(fragment.graph.V)]
                            ))
        if known_atom_mappings is not None:
            reduced_mappings = []
            known_lhs_indices = [x[0] for x in known_atom_mappings]
            for m in mappings:
                bad_pairing = False
                for pairing in m[0]:
                    if pairing[0] in known_lhs_indices and pairing not in known_atom_mappings:
                        bad_pairing = True
                        break
                if not bad_pairing:
                    reduced_mappings.append(m)
            return reduced_mappings
        return mappings

    # Start building a mapping from fragments to the 'continuous-LHS'
    #  1. Pick (first) largest fragment from all lhs/rhs combinations
    #  2. Remove all atoms that are now matched on the LHS and RHS
    #  3. Reevaluate the best matched for all LHS/RHS fragment combinations
    #  4. Repeat 1./2. until no fragments and/or 'continuous-LHS' indices are free
    remaining_mappings = get_mappings()
    starters = []
    # TODO try multiple times, here? (loop)
    starters.append(max(remaining_mappings, key=lambda x: len(x[0])))
    used = []
    l_blocked: Set[int] = set()
    r_blocked: Set[int] = set()
    while len(remaining_mappings) > 0:
        current = max(remaining_mappings, key=lambda x: len(x[0]))
        used.append(current)
        assert not bool(l_blocked & current[1])
        assert not bool(r_blocked & current[2])
        l_blocked.update(current[1])
        r_blocked.update(current[2])

        # Reduce origin fragments
        lhs_origin = current[4]
        rhs_origin = current[5]
        lhs_origin_fragment = remaining_lhs_fragments.pop(lhs_origin)
        rhs_origin_fragment = remaining_rhs_fragments.pop(rhs_origin)
        lhs_origin_indices = remaining_lhs_indices.pop(lhs_origin)
        rhs_origin_indices = remaining_rhs_indices.pop(rhs_origin)
        lhs_indices_by_subgraph = sort_indices_by_subgraph(
            [i for i in range(lhs_origin_fragment.graph.V) if i not in current[6]],
            lhs_origin_fragment.graph
        )
        for subgraph_indices in lhs_indices_by_subgraph:
            mol, original_indices = mol_from_subgraph_indices(subgraph_indices, copy(lhs_origin_fragment))
            if mol:
                remaining_lhs_fragments.append(mol)
                remaining_lhs_indices.append([lhs_origin_indices[x] for x in original_indices])
        rhs_indices_by_subgraph = sort_indices_by_subgraph(
            [i for i in range(rhs_origin_fragment.graph.V) if i not in current[7]],
            rhs_origin_fragment.graph
        )
        for subgraph_indices in rhs_indices_by_subgraph:
            mol, original_indices = mol_from_subgraph_indices(subgraph_indices, copy(rhs_origin_fragment))
            if mol:
                remaining_rhs_fragments.append(mol)
                remaining_rhs_indices.append([rhs_origin_indices[x] for x in original_indices])
        remaining_mappings = get_mappings()

        def is_overlapping(fragment_info):
            return bool(set(l_blocked) & set(fragment_info[1])) or bool(set(r_blocked) & set(fragment_info[2]))
        remaining_mappings[:] = [x for x in remaining_mappings if not is_overlapping(x)]
    assert len(l_blocked) == n_atoms

    # Prepare the final output, for each atom in each molecule, generate
    #  a tuple pointing at a molecule and an atom within it on the other
    #  side.
    all_maps: List[Tuple[int, int]] = []
    for u in used:
        all_maps += u[0]

    def continuous_to_fragments(index: int, offsets: List[int]) -> Tuple[int, int]:
        for o in reversed(offsets):
            if o <= index:
                return (offsets.index(o), index-o)
        raise RuntimeError('Bug: Offsets in `map_reaction_from_molecules` did not match.')

    all_maps.sort(key=lambda x: x[0])
    lhs_final: List[List[Tuple[int, int]]] = []
    for i, l_mol in enumerate(lhs):
        tmp: List[Tuple[int, int]] = []
        for j in range(l_mol.graph.V):
            tmp.append(continuous_to_fragments(all_maps[j+l_offsets[i]][1], r_offsets))
        lhs_final.append(tmp)
    all_maps.sort(key=lambda x: x[1])
    rhs_final: List[List[Tuple[int, int]]] = []
    for i, r_mol in enumerate(rhs):
        tmp = []
        for j in range(r_mol.graph.V):
            tmp.append(continuous_to_fragments(all_maps[j+r_offsets[i]][0], l_offsets))
            assert lhs_final[tmp[-1][0]][tmp[-1][1]][0] == i
            assert lhs_final[tmp[-1][0]][tmp[-1][1]][1] == j
        rhs_final.append(tmp)

    return lhs_final, rhs_final


def map_reaction_from_molecules_cached(
    lhs: List[masm.Molecule],
    rhs: List[masm.Molecule],
    known_atom_mappings: Optional[List[Tuple[int, int]]] = None
) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
    """Tries to match each atom in the molecules of the left-hand side to
    exactly one atom on the right-hand side.

    Atom type counts must match.

    Note
    ----

    The algorithm is experimental, be cautious with its results.
    This version of the algorithm is memory greedy.

    Parameters
    ----------
    lhs : List[masm.Molecule]
        A list of molecules on the left-hand side of the matching.
    rhs : List[masm.Molecule]
        A list of molecules on the right-hand side of the matching.
    known_atom_mappings : Optional[List[Tuple[int, int]]]
        A list of atom mappings that have to be respected. The atoms are
        indexed on a continuous scale in order of the atoms in the molecules
        given.

    Returns
    -------
    Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]
        A tuple atom matches. The first entry matching from left- to right-hand
        side the second from right- to left-hand side.

    Raises
    ------
    RuntimeError
        If algorithm fails or miss matches appear.
    """
    n_atoms = sum([lmol.graph.V for lmol in lhs])
    assert sum([rmol.graph.V for rmol in lhs]) == n_atoms
    # In this algorithm we will try to rebuild the LHS from fragments of
    #  the rhs. To this end the LHS will be viewed as having a continuous
    #  index with breaks where a molecule ends and a new one starts.
    #  (e.g. the input methane, HCl and benzene, in that order, would
    #  correspond to [0,1,2,3,4 | 5,6 | 7,8,9,10,11,...,18]
    # When generating the intersection based fragment mapping we will generate
    #  the mapping to this new 'continuous-LHS' index.
    l_sizes = [m.graph.V for m in lhs]
    r_sizes = [m.graph.V for m in rhs]
    l_offsets: List[int] = [sum(l_sizes[:i]) for i in range(len(lhs))]
    r_offsets: List[int] = [sum(r_sizes[:i]) for i in range(len(rhs))]
    mappings = []
    for i, l_mol in enumerate(lhs):
        for j, r_mol in enumerate(rhs):
            fragments, fragments_to_l_mol, fragments_to_r_mol = all_matching_fragments(
                l_mol, r_mol, min_fragment_size=1
            )
            # Generate continuous-LHS mapping, and sets of the mapped indices
            for fragment, l_maps, r_maps in zip(fragments, fragments_to_l_mol, fragments_to_r_mol):
                for l_map in l_maps:
                    assert len(l_map) == fragment.graph.V
                    for r_map in r_maps:
                        assert len(r_map) == fragment.graph.V
                        mapping = [(l_map[x]+l_offsets[i], r_map[x]+r_offsets[j]) for x in range(fragment.graph.V)]
                        l_set = set([x[0] for x in mapping])
                        r_set = set([x[1] for x in mapping])
                        mappings.append((mapping, l_set, r_set, fragment))
    # Start building a mapping from fragments to the 'continuous-LHS'
    #  1. Pick (first) largest fragment
    #  2. Remove all fragments that overlap with it or the now blocked
    #     indices in the 'continuous-LHS' indices
    #  3. Repeat 1. until no fragments and/or 'continuous-LHS' indices are free
    if known_atom_mappings is not None:
        remaining_mappings = []
        known_lhs_indices = [x[0] for x in known_atom_mappings]
        for m in mappings:
            bad_pairing = False
            for pairing in m[0]:
                if pairing[0] in known_lhs_indices and pairing not in known_atom_mappings:
                    bad_pairing = True
                    break
            if not bad_pairing:
                remaining_mappings.append(m)
    else:
        remaining_mappings = copy(mappings)
    starters = []
    # TODO try multiple times, here? (loop)
    starters.append(max(remaining_mappings, key=lambda x: len(x[0])))
    used = []
    l_blocked: Set[int] = set()
    r_blocked: Set[int] = set()
    while len(remaining_mappings) > 0:
        current = max(remaining_mappings, key=lambda x: len(x[0]))
        used.append(current)
        assert not bool(l_blocked & current[1])
        assert not bool(r_blocked & current[2])
        l_blocked.update(current[1])
        r_blocked.update(current[2])

        def is_overlapping(fragment_info):
            return bool(set(l_blocked) & set(fragment_info[1])) or bool(set(r_blocked) & set(fragment_info[2]))
        remaining_mappings[:] = [x for x in remaining_mappings if not is_overlapping(x)]
    if len(l_blocked) != n_atoms:
        raise RuntimeError(
            'Could not complete atom mapping across reaction. Check "known_atom_mappings" if any were given.'
        )

    # Prepare the final output, for each atom in each molecule, generate
    #  a tuple pointing at a molecule and an atom within it on the other
    #  side.
    all_maps: List[Tuple[int, int]] = []
    for u in used:
        all_maps += u[0]

    def continuous_to_fragments(index: int, offsets: List[int]) -> Tuple[int, int]:
        for o in reversed(offsets):
            if o <= index:
                return (offsets.index(o), index-o)
        raise RuntimeError('Bug: Offsets in `map_reaction_from_molecules` did not match.')

    all_maps.sort(key=lambda x: x[0])
    lhs_final: List[List[Tuple[int, int]]] = []
    for i, l_mol in enumerate(lhs):
        tmp: List[Tuple[int, int]] = []
        for j in range(l_mol.graph.V):
            tmp.append(continuous_to_fragments(all_maps[j+l_offsets[i]][1], r_offsets))
        lhs_final.append(tmp)
    all_maps.sort(key=lambda x: x[1])
    rhs_final: List[List[Tuple[int, int]]] = []
    for i, r_mol in enumerate(rhs):
        tmp = []
        for j in range(r_mol.graph.V):
            tmp.append(continuous_to_fragments(all_maps[j+r_offsets[i]][0], l_offsets))
            assert lhs_final[tmp[-1][0]][tmp[-1][1]][0] == i
            assert lhs_final[tmp[-1][0]][tmp[-1][1]][1] == j
        rhs_final.append(tmp)

    return lhs_final, rhs_final
