#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


import scine_molassembler as masm
from copy import copy, deepcopy

from typing import List, Tuple, Optional, Set, Dict


def get_atom_mapping(m1: masm.Molecule, m2: masm.Molecule) -> List[int]:
    """Maps atoms between two identical molecules.

    Parameters
    ----------
    m1 : scine_molassembler.Molecule
        The first molecule.
    m2 : scine_molassembler.Molecule
        The second molecule.

    Returns
    -------
    List[int]
        Mapping of atom indices of the first molecule to those of
        the second molecule.

    Raises
    ------
    RuntimeError
        If molecules are not identical.

    Examples
    --------
    >>> mapping = get_atom_mapping(m1, m2)
    >>> for i in range(m1.graph.V):
    >>>     assert m1.graph.element_type() == m2.graph.element_type(mapping[i])

    """
    if m1 != m2:
        raise RuntimeError("Function get_atom_mapping expected identical molecules.")
    c1 = copy(m1)
    c2 = copy(m2)
    map1 = c1.canonicalize()
    map2 = c2.canonicalize()
    return [map2.index(i) for i in map1]


def set_difference(
    m1: masm.Molecule, m2: masm.Molecule
) -> Tuple[List[masm.Molecule], List[List[List[int]]]]:
    """_summary_

    Parameters
    ----------
    m1 : scine_molassembler.Molecule
        _description_
    m2 : scine_molassembler.Molecule
        _description_

    Returns
    -------
    Tuple[List[scine_molassembler.Molecule], List[List[List[int]]]]
        _description_
    """
    matching_ids_list = matching_atom_ids(m1, m2)
    molecular_fragments: List[masm.Molecule] = []
    index_mapping: List[List[List[int]]] = []
    for matching_ids in matching_ids_list:
        m1_set_diff_idxs = [i for i in range(m1.graph.V) if i not in matching_ids[0]]
        m1_indices_by_subgraph = sort_indices_by_subgraph(m1_set_diff_idxs, m1.graph)
        for subgraph_indices in m1_indices_by_subgraph:
            mol, original_indices = mol_from_subgraph_indices(subgraph_indices, m1)
            if mol is not None:
                map = mol.canonicalize()
                final_indices = [original_indices[map.index(i)] for i in range(len(map))]
                if mol not in molecular_fragments:
                    molecular_fragments.append(mol)
                    index_mapping.append([final_indices])
                else:
                    idx = molecular_fragments.index(mol)
                    index_mapping[idx].append(final_indices)
    return molecular_fragments, index_mapping


def symmetric_difference(
    m1: masm.Molecule, m2: masm.Molecule
) -> Tuple[List[masm.Molecule], List[Tuple[List[List[int]], List[List[int]]]]]:
    """_summary_

    Parameters
    ----------
    m1 : scine_molassembler.Molecule
        _description_
    m2 : scine_molassembler.Molecule
        _description_

    Returns
    -------
    Tuple[List[scine_molassembler.Molecule], List[Tuple[List[List[int]], List[List[int]]]]]
        _description_
    """
    matching_ids_list = matching_atom_ids(m1, m2)
    molecular_fragments: List[masm.Molecule] = []
    index_mapping: List[Tuple[List[List[int]], List[List[int]]]] = []
    for matching_ids in matching_ids_list:
        m1_set_diff_idxs = [i for i in range(m1.graph.V) if i not in matching_ids[0]]
        m1_indices_by_subgraph = sort_indices_by_subgraph(m1_set_diff_idxs, m1.graph)
        for subgraph_indices in m1_indices_by_subgraph:
            mol, original_indices = mol_from_subgraph_indices(subgraph_indices, m1)
            if mol is not None:
                map = mol.canonicalize()
                final_indices = [original_indices[map.index(i)] for i in range(len(map))]
                if mol not in molecular_fragments:
                    molecular_fragments.append(mol)
                    index_mapping.append(([final_indices], []))
                else:
                    idx = molecular_fragments.index(mol)
                    index_mapping[idx][0].append(final_indices)
        m2_set_diff_idxs = [i for i in range(m2.graph.V) if i not in matching_ids[1]]
        m2_indices_by_subgraph = sort_indices_by_subgraph(m2_set_diff_idxs, m2.graph)
        for subgraph_indices in m2_indices_by_subgraph:
            mol, original_indices = mol_from_subgraph_indices(subgraph_indices, m2)
            if mol is not None:
                map = mol.canonicalize()
                final_indices = [original_indices[map.index(i)] for i in range(len(map))]
                if mol not in molecular_fragments:
                    molecular_fragments.append(mol)
                    index_mapping.append(([], [final_indices]))
                else:
                    idx = molecular_fragments.index(mol)
                    index_mapping[idx][1].append(final_indices)
    return molecular_fragments, index_mapping


def intersection(
    m1: masm.Molecule, m2: masm.Molecule
) -> Tuple[List[masm.Molecule], List[Tuple[List[List[int]], List[List[int]]]]]:
    """Identifies all intersecting fragments in two molecules.

    Parameters
    ----------
    m1 : scine_molassembler.Molecule
        The first molecule.
    m2 : scine_molassembler.Molecule
        The second molecule.

    Returns
    -------
    Tuple[List[scine_molassembler.Molecule], List[Tuple[List[List[int]], List[List[int]]]]]
        A list of unique molecular fragments that are present in both
        molecules.
        Additionally, for each fragment all mappings of atom indices to each of
        the two molecules are returned.

    Examples
    --------
    >>> for fragment, mappings in intersection(m1, m2):
    >>>     print(mappings[0]) #  all mappings of fragment in m1
    >>>     print(mappings[1]) #  all mappings of fragment in m2

    """
    fragments, m1_mappings, m2_mappings = all_matching_fragments(m1, m2)
    return fragments, list(zip(m1_mappings, m2_mappings))


def mol_from_subgraph_indices(
    subgraph_indices: List[int], molecule: masm.Molecule
) -> Tuple[Optional[masm.Molecule], List[int]]:
    """Cuts down molecule to fragment identified by atom
    indices.

    Additionally produces direct atom mapping between the
    initial molecule and the resulting fragment.

    Parameters
    ----------
    subgraph_indices : List[int]
        Atom indices of a connected subgraph in the given molecule.
    molecule : scine_molassembler.Molecule
        The molecule to reduce into a fragment.

    Returns
    -------
    Tuple[Optional[scine_molassembler.Molecule], List[int]]
        The molecular fragment and its atom mapping towards
        the original molecule. The molecule returned will be
        ``None`` if the initial index list was empty.
    """
    submol = copy(molecule)
    n = len(subgraph_indices)
    assert len(set(subgraph_indices)) == n
    for idx in subgraph_indices:
        assert submol.graph.V > idx
    if n == 0:
        return None, []
    elif n == submol.graph.V:
        return submol, list(range(n))
    else:
        original_indices = list(range(submol.graph.V))
        indices_to_be_removed = [i for i in range(submol.graph.V) if i not in subgraph_indices]
        while indices_to_be_removed:
            previous_number = len(indices_to_be_removed)
            for i in reversed(range(len(indices_to_be_removed))):
                index = indices_to_be_removed[i]
                if submol.graph.can_remove(index):
                    c = deepcopy(submol)
                    try:
                        submol.remove_atom(index)
                    except RuntimeError:
                        submol = c
                        continue
                    indices_to_be_removed.pop(i)
                    original_indices.pop(index)
                    for j, other in enumerate(indices_to_be_removed):
                        if index < other:
                            indices_to_be_removed[j] -= 1
            if previous_number == len(indices_to_be_removed):
                raise RuntimeError("Molassembler bug, could not deconstruct graph.")
    return submol, original_indices


def sort_indices_by_subgraph(ids: List[int], graph: masm.Graph) -> List[List[int]]:
    """Groups the given indices by subgraphs.

    All indices that are directly connected to one another
    (connected in a single subgraph) are sorted together.
    All indices will be returned, in a limiting case
    each index will be part of its own subgraph.

    Parameters
    ----------
    ids : List[int]
        Indices to be sorted.
    graph : scine_molassembler.Graph
        Molecular graph to sort the indices by.

    Returns
    -------
    List[List[int]]
        A list of subgraphs, each given as list of indices.
    """
    if not ids:
        return []
    assert max(ids) < graph.V
    subgraphs: List[List[int]] = [[ids.pop(0)]]
    for id in ids:
        assert id < graph.V
        adjacents = []
        for i_sg, subgraph in enumerate(subgraphs):
            for atom in subgraph:
                if graph.adjacent(id, atom):
                    adjacents.append(i_sg)
                    break
        n = len(adjacents)
        if n == 0:
            subgraphs.append([id])
        elif n == 1:
            subgraphs[adjacents[0]].append(id)
        else:
            subgraphs[adjacents[0]].append(id)
            # join graphs
            for i in reversed(adjacents[1:]):
                subgraphs[adjacents[0]] += subgraphs.pop(i)
    # sort all subgraphs
    subgraphs.sort()
    return subgraphs


def is_single_subgraph(ids: List[int], graph: masm.Graph) -> bool:
    """Checks if the set of indices given describes a
    single, connected subgraph/fragment of the given molecular
    graph.

    Parameters
    ----------
    ids : List[int]
        List of atom indices.
    graph : scine_molassembler.Graph
        Molecular graph.

    Returns
    -------
    bool
        True if the indices describe one connected subgraph.
        False otherwise.
    """
    for id1 in reversed(ids):
        assert id1 < graph.V
        adj = [a for a in graph.adjacents(id1)]
        for id2 in reversed(ids):
            if id2 in adj:
                break
        else:
            return False
    return True


def matching_atom_ids(
    m1: masm.Molecule, m2: masm.Molecule
) -> List[Tuple[Set[int], Set[int]]]:
    """Generates sets of atom indices of all
    matching molecular fragments.

    Parameters
    ----------
    m1 : scine_molassembler.Molecule
        The first molecule.
    m2 : scine_molassembler.Molecule
        The second molecule.

    Returns
    -------
    List[Tuple[Set[int], Set[int]]]
        A list of pairs of atom index sets.
        The sets represent atom indices in the respective
        molecules. Each pair represents one matching fragment.
        Fragments are non-unique, as a single fragment of one
        molecule may match multiple in the other.
    """
    swapped = False
    if m1.graph.V > m2.graph.V:
        swapped = True
        tmp = m1
        m1 = m2
        m2 = tmp
    complete_matches = masm.subgraphs.complete(m1, m2)
    if len(complete_matches) > 0:
        unique_m1_indices = []
        unique_m2_indices = []
        m1_range = set(range(m1.graph.V))
        for match in complete_matches:
            unique_m1_indices.append(m1_range)
            unique_m2_indices.append(set([i for i, _ in match.right]))
            assert m1.graph.V == len(unique_m2_indices[-1])
        if swapped:
            return list(zip(unique_m2_indices, unique_m1_indices))
        else:
            return list(zip(unique_m1_indices, unique_m2_indices))
    else:
        m1_indices, m2_indices = __match_recursive(m1, m2)
        m1_indices, m2_indices = [
            list(t) for t in zip(*sorted(zip(m1_indices, m2_indices), reverse=True, key=lambda x: len(x[0])))
        ]
        assert len(m2_indices) == len(m1_indices)
        unique_m2_indices = []
        unique_m1_indices = []
        if m2_indices:
            unique_m2_indices = [m2_indices[0]]
            unique_m1_indices = [m1_indices[0]]
            for m2_set, m1_set in zip(m2_indices[1:], m1_indices[1:]):
                for u_m1_set, u_m2_set in zip(unique_m1_indices, unique_m2_indices):
                    if (m1_set.issubset(u_m1_set) and m2_set.issubset(u_m2_set)):
                        break
                else:
                    unique_m1_indices.append(m1_set)
                    unique_m2_indices.append(m2_set)
        if swapped:
            return list(zip(unique_m2_indices, unique_m1_indices))
        else:
            return list(zip(unique_m1_indices, unique_m2_indices))


def __match_recursive(
        m1: masm.Molecule,
        m2: masm.Molecule,
        m1_indices: Optional[List[int]] = None,
        m1_index_cache: Optional[List[List[int]]] = None
) -> Tuple[List[Set[int]], List[Set[int]]]:
    """Recursively deconstructs molecule one to find its
    fragments in molecule two.

    Note
    ----
    For best performance the size of the first molecule
    should be smaller than that of the second one.

    Parameters
    ----------
    m1 : scine_molassembler.Molecule
        The first molecule (should be the smaller one).
    m2 : scine_molassembler.Molecule
        The second molecule.
    m1_indices : Optional[List[int]], optional
        Cache for remaining original indices of molecule one,
        by default None
    m1_index_cache : Optional[List[List[int]]], optional
        Cache for known maximal size matching fragments,
        identified by indices of atoms in the original
        version of molecule one. By default None

    Returns
    -------
    Tuple[List[Set[int]], List[Set[int]]]
        Two lists of atom index sets, each index set describes
        one fragment of a molecule. Sets at the same positions
        in the two lists identify the same fragments in both
        molecule one and molecule two. The indices in the first
        list are to be applied to the first molecule the ones
        in the second to the second molecule.
    """
    matching_m2_indices = []
    matching_m1_indices = []
    reduced_molecules = []
    reduced_m1_index_cache = []
    if m1_index_cache is None:
        m1_index_cache = []
    if m1_indices is None:
        m1_indices = list(range(m1.graph.V))
    if m1.graph.V == 2:
        return ([], [])
    for i in range(m1.graph.V):
        new_molecule = copy(m1)
        new_m1_indices = copy(m1_indices)
        if not new_molecule.graph.can_remove(i):
            continue
        new_molecule.remove_atom(i)
        new_m1_indices.pop(i)
        new_matches = masm.subgraphs.complete(new_molecule, m2)
        new_m2_matches = [set([i for i, _ in match.right]) for match in new_matches]
        matching_m2_indices += new_m2_matches
        for _ in new_m2_matches:
            matching_m1_indices.append(set(new_m1_indices))
        reduced_molecules.append(new_molecule)
        reduced_m1_index_cache.append(new_m1_indices)
    if len(matching_m2_indices) == 0:
        for mol, indices in zip(reduced_molecules, reduced_m1_index_cache):
            for cache in m1_index_cache:
                # Screen if another recursion has already run through
                #  this branch by comparing m1 indices with the cache
                if set(indices).issubset(cache):
                    break
            else:
                m1_index_cache.append(indices)
                m1_results, m2_results = __match_recursive(
                    mol, m2, m1_indices=indices, m1_index_cache=m1_index_cache)
                matching_m1_indices += m1_results
                matching_m2_indices += m2_results
    return matching_m1_indices, matching_m2_indices


def maximum_matching_fragments(
    m1: masm.Molecule, m2: masm.Molecule, min_fragment_size: int = 2
) -> Tuple[List[masm.Molecule], List[List[List[int]]], List[List[List[int]]]]:
    swapped = False
    if m1.graph.V > m2.graph.V:
        swapped = True
        tmp = m1
        m1 = m2
        m2 = tmp
    complete_matches = masm.subgraphs.complete(m1, m2)
    if len(complete_matches) > 0:
        unique_m1_indices = []
        unique_m2_indices = []
        m1_range = list(range(m1.graph.V))
        for match in complete_matches:
            unique_m1_indices.append(m1_range)
            unique_m2_indices.append([i for _, i in match.left])
            assert m1.graph.V == len(unique_m2_indices[-1])
        if swapped:
            return [m1], [unique_m2_indices], [unique_m1_indices]
        else:
            return [m1], [unique_m1_indices], [unique_m2_indices]
    else:
        fragments, m1_indices, m2_indices = __fragment_recursion(
            m1,
            m2,
            min_fragment_size=min_fragment_size
        )
        assert len(fragments) == len(m1_indices)
        assert len(m2_indices) == len(m1_indices)
        if swapped:
            return fragments, m2_indices, m1_indices
        else:
            return fragments, m1_indices, m2_indices


def all_matching_fragments(
    m1: masm.Molecule, m2: masm.Molecule, min_fragment_size: int = 2
) -> Tuple[List[masm.Molecule], List[List[List[int]]], List[List[List[int]]]]:
    swapped = False
    if m1.graph.V > m2.graph.V:
        swapped = True
        tmp = m1
        m1 = m2
        m2 = tmp
    complete_matches = masm.subgraphs.complete(m1, m2)
    if len(complete_matches) > 0:
        unique_m1_indices: List[List[int]] = []
        unique_m2_indices: List[List[int]] = []
        m1_range = list(range(m1.graph.V))
        unique_m1_indices.append(m1_range)
        for match in complete_matches:
            unique_m2_indices.append([i for _, i in match.left])
            assert m1.graph.V == len(unique_m2_indices[-1])
        reduced_molecule_cache = {}
        reduced_molecule_cache[m1.graph.__repr__().split()[-1]] = [m1]
        fragments, m1_indices, m2_indices = __fragment_recursion(
            m1,
            m2,
            find_all=True,
            min_fragment_size=min_fragment_size,
            reduced_molecule_cache=reduced_molecule_cache,
            matching_m2_index_cache=deepcopy(unique_m2_indices)
        )
        if swapped:
            return [m1]+fragments, [unique_m2_indices]+m2_indices, [unique_m1_indices]+m1_indices
        else:
            return [m1]+fragments, [unique_m1_indices]+m1_indices, [unique_m2_indices]+m2_indices
    else:
        fragments, m1_indices, m2_indices = __fragment_recursion(
            m1,
            m2,
            find_all=True,
            min_fragment_size=min_fragment_size
        )
        assert len(fragments) == len(m1_indices)
        assert len(m2_indices) == len(m1_indices)
        if swapped:
            return fragments, m2_indices, m1_indices
        else:
            return fragments, m1_indices, m2_indices


def __fragment_recursion(
        m1: masm.Molecule,
        m2: masm.Molecule,
        find_all: bool = False,
        min_fragment_size: int = 2,
        m1_indices: Optional[List[int]] = None,
        reduced_molecule_cache: Optional[Dict[str, List[masm.Molecule]]] = None,
        matching_m2_index_cache: Optional[List[List[int]]] = None,
        full_m1: Optional[masm.Molecule] = None
) -> Tuple[List[masm.Molecule], List[List[List[int]]], List[List[List[int]]]]:
    if matching_m2_index_cache is None:
        matching_m2_index_cache = []
    if full_m1 is None:
        full_m1 = copy(m1)
    matching_m2_indices: List[List[List[int]]] = []
    matching_m1_indices: List[List[List[int]]] = []
    reduced_molecules = []
    reduced_m1_indices = []
    matching_fragments: List[masm.Molecule] = []
    if reduced_molecule_cache is None:
        reduced_molecule_cache = {}
    if m1_indices is None:
        m1_indices = list(range(m1.graph.V))
    assert min_fragment_size > 0
    if (m1.graph.V-1) < min_fragment_size:
        return [], [], []
    for i in range(m1.graph.V):
        new_molecule = copy(m1)
        new_m1_indices = copy(m1_indices)
        if not new_molecule.graph.can_remove(i):
            continue
        new_molecule.remove_atom(i)
        new_m1_indices.pop(i)
        # TODO is a canonicalized cache faster?
        # new_canonical = copy(new_molecule)
        # _ = new_canonical.canonicalize()
        sum_formula = new_molecule.graph.__repr__().split()[-1]
        if sum_formula in reduced_molecule_cache:
            if new_molecule in reduced_molecule_cache[sum_formula]:
                continue
        else:
            reduced_molecule_cache[sum_formula] = []
        if new_molecule.graph.E <= m2.graph.E and len(new_molecule.graph.cycles) <= len(m2.graph.cycles):
            new_m2_matches = masm_complete_graph_match(new_molecule, m2)
        else:
            new_m2_matches = []
        if new_m2_matches:
            matching_fragments.append(new_molecule)
            matching_m2_indices.append(new_m2_matches)
            matching_m2_index_cache.append(new_m2_matches)
            new_m1_matches = masm_complete_graph_match(new_molecule, full_m1)
            matching_m1_indices.append(new_m1_matches)
        reduced_molecules.append(new_molecule)
        reduced_m1_indices.append(new_m1_indices)
        reduced_molecule_cache[sum_formula].append(new_molecule)
    if len(matching_m2_indices) == 0 or find_all:
        for mol, indices in zip(reduced_molecules, reduced_m1_indices):
            mol_results, m1_results, m2_results = __fragment_recursion(
                mol,
                m2,
                find_all=find_all,
                min_fragment_size=min_fragment_size,
                m1_indices=indices,
                reduced_molecule_cache=reduced_molecule_cache,
                matching_m2_index_cache=matching_m2_index_cache,
                full_m1=full_m1
            )
            if not find_all and mol_results:
                min_fragment_size = max(mol_results[0].graph.V, min_fragment_size)
            matching_fragments += mol_results
            matching_m1_indices += m1_results
            matching_m2_indices += m2_results
        if not find_all and matching_fragments:
            for i in reversed(range(len(matching_fragments))):
                if matching_fragments[i].graph.V < min_fragment_size:
                    matching_fragments.pop(i)
                    matching_m1_indices.pop(i)
                    matching_m2_indices.pop(i)
    return matching_fragments, matching_m1_indices, matching_m2_indices


def masm_complete_graph_match(m1: masm.Molecule, m2: masm.Molecule):
    if m1.graph.V > m2.graph.V:
        return []
    if m1.graph.V == 1:
        atom_type = m1.graph.element_type(0)
        return [[i] for i in m2.graph.atoms_of_element(atom_type)]
    matches = masm.subgraphs.complete(m1, m2)
    return [[x for _, x in match.left] for match in matches]
