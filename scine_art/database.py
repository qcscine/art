#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


import scine_molassembler as masm

from scine_art.reaction_template import ReactionTemplate, Match
from scine_art.io import write_molecule_to_svg


from typing import List, Optional, Any, Generator
from copy import deepcopy
import pickle
import svgutils
import os


class ReactionTemplateDatabase:
    """A database of reaction templates."""
    def __init__(self) -> None:
        self.__unique_templates: List[ReactionTemplate] = []

    def get_template(self, uuid: str) -> Optional[ReactionTemplate]:
        """Getter for a single reaction template.

        Parameters
        ----------
        uuid : str
            The unique ID of a reaction template.

        Returns
        -------
        Optional[ReactionTemplate]
            The reaction template with the given unique ID if present.
        """
        for template in self.__unique_templates:
            if template.get_uuid() == uuid:
                return deepcopy(template)
        return None

    def iterate_templates(self) -> Generator[ReactionTemplate, None, None]:
        """Iterator through all stored reaction templates.

        Yields
        ------
        Generator[ReactionTemplate, None, None]
            Iterator of all reaction templates in the database.
        """
        for template in self.__unique_templates:
            yield deepcopy(template)

    def add_template(
        self,
        new_template: ReactionTemplate
    ) -> str:
        """Add a new template to the database.

        Will deduplicate the reaction template against stored ones.

        Parameters
        ----------
        new_template : ReactionTemplate
            The new reaction template

        Returns
        -------
        str
            Returns the UUID of the now stored template, if the template
            was a duplicate and thus subsumed, the UUID of the existing match
            is returned.
        """
        for template in self.__unique_templates:
            if template == new_template:
                template.update(new_template)
                return template.get_uuid()
        self.__unique_templates.append(new_template)
        return new_template.get_uuid()

    def find_matching_templates(
        self,
        molecules: List[masm.Molecule],
        energy_cutoff: Optional[float] = 300,
        enforce_atom_shapes: bool = True
    ) -> Optional[List[Match]]:
        """Finds all matching reaction templates for a given set of molecules.

        All given molecules must be used in the matched reaction template
        exactly once. None may be unused or used twice.

        A single template can match multiple atom combinations in the given
        atoms, hence multiple matches can be returned per reaction template.

        Note
        ----

        For a more fine grained matching algorithm loop over all individual
        templates and use their matching methods which allow for more optional
        arguments.

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

        Returns
        -------
        Optional[List[Match]]
            If any matches exist, returns a list of all possible matches.
        """
        results = []
        for template in self.__unique_templates:
            if energy_cutoff is None:
                matches = template.determine_all_matches(
                    molecules,
                    energy_cutoff=0.0,
                    enforce_atom_shapes=enforce_atom_shapes,
                    enforce_lhs=True,
                    enforce_rhs=True,
                )
            else:
                matches = template.determine_all_matches(
                    molecules,
                    energy_cutoff=energy_cutoff,
                    enforce_atom_shapes=enforce_atom_shapes
                )
            if matches is not None:
                results += matches
        if not results:
            return None
        else:
            return results

    def save(self, path: str = '.rtdb.pickle.obj') -> None:
        """Saves the currently stored unique reaction templates into a file.

        Parameters
        ----------
        path : str, optional
            The path to store a list of reaction templates, by default
            ``'.rtdb.pickle.obj'``
        """
        with open(path, 'wb') as f:
            pickle.dump(self.__unique_templates, f)

    def load(self, path: str = '.rtdb.pickle.obj') -> None:
        """Loads a stored (pickled) file of templates and replaces existing
        data.

        Parameters
        ----------
        path : str, optional
            The path to a stored list of reaction templates, by default
            ``'.rtdb.pickle.obj'``
        """
        with open(path, 'rb') as f:
            self.__unique_templates = pickle.load(f)

    def append_file(self, path: str = '.rtdb.pickle.obj') -> None:
        """Loads a stored (pickled) file of templates and adds it to the
        existing unique reaction templates, keeps current template and adds data
        from the file.

        Parameters
        ----------
        path : str, optional
            The path to a stored list of reaction templates, by default
            ``'.rtdb.pickle.obj'``
        """
        with open(path, 'rb') as f:
            new_templates = pickle.load(f)
        for template in new_templates:
            self.add_template(template)

    def template_count(self) -> int:
        """Getter for the number of reaction templates int he database.

        Returns
        -------
        int
            Number of unique reaction templates in the database.
        """
        return len(self.__unique_templates)

    # TODO requires masm.Molecule JSON representation generation on the fly
    # def dump_json(self, path: str = 'rtdb.dump.json') -> None:
    #     with open(path, 'w') as f:
    #         for template in self.__unique_templates:
    #             f.write(template.to_json())

    def add_scine_database(
        self,
        host: str = 'localhost',
        port: int = 27017,
        name: str = 'default',
        energy_cutoff: float = 300,
        model: Optional[Any] = None,
        gibbs_free_energy: bool = True
    ) -> None:
        """Parse an entire SCINE Database adding and deduplicating all templates
        into this database.

        Parameters
        ----------
        host : str, optional
            The IP or hostname of the database server, by default 'localhost'
        port : int, optional
            The port of the database server, by default 27017
        name : str, optional
            The name of the database on the server, by default 'default'
        energy_cutoff : float, optional
            Only store templates where at least one recorded barrier of the
            reaction is below the given threshold (in kJ/mol), by default
            300 kJ/mol.
        model : Optional[scine_database.Model], optional
            The scine_database.Model to be allowed for energy evaluations, by
            default ``None`` is given, data of all models in the SCINE Database
            are allowed.
        gibbs_free_energy : bool, optional
            If true will only consider Gibbs free energies as valid energies
            for evaluation, by default True
        """
        import scine_database as db

        manager = db.Manager()
        credentials = db.Credentials(host, port, name)
        manager.set_credentials(credentials)
        manager.connect()
        reaction_collection = manager.get_collection('reactions')
        elementary_step_collection = manager.get_collection('elementary_steps')
        property_collection = manager.get_collection('properties')
        structure_collection = manager.get_collection('structures')
        if gibbs_free_energy:
            energy_label = 'gibbs_free_energy'
        else:
            energy_label = 'electronic_energy'

        def get_energy_for_structure(
                structure: db.Structure,
                model: db.Model,
        ) -> Optional[float]:
            structure.link(structure_collection)
            structure_properties = structure.query_properties(energy_label, model, property_collection)
            if len(structure_properties) < 1:
                return None
            # pick last property if multiple
            prop = db.NumberProperty(structure_properties[-1])
            prop.link(property_collection)
            return prop.get_data()

        for reaction in reaction_collection.iterate_all_reactions():
            reaction.link(reaction_collection)
            elementary_steps = reaction.get_elementary_steps()
            best_spline = None
            best_step = None
            best_barriers = (9999999.9, 9999999.9)
            best_ts = 9999999.9
            for step_id in elementary_steps:
                step = db.ElementaryStep(step_id)
                step.link(elementary_step_collection)
                if step.has_spline():
                    spline = step.get_spline()
                    ts = db.Structure(step.get_transition_state())
                    ts.link(structure_collection)
                    current_model = model
                    if current_model is None:
                        current_model = ts.get_model()
                    e_ts = get_energy_for_structure(
                        ts,
                        current_model
                    )
                    if e_ts is None:
                        continue
                    missing_energy = False
                    e_lhs = 0.0
                    for reactant in step.get_reactants(db.Side.LHS)[0]:
                        energy = get_energy_for_structure(
                            db.Structure(reactant),
                            current_model
                        )
                        if energy is None:
                            missing_energy = True
                            break
                        e_lhs += energy
                    e_rhs = 0.0
                    for reactant in step.get_reactants(db.Side.RHS)[1]:
                        energy = get_energy_for_structure(
                            db.Structure(reactant),
                            current_model
                        )
                        if energy is None:
                            missing_energy = True
                            break
                        e_rhs += energy
                    if missing_energy:
                        continue

                    forward_barrier = (e_ts-e_lhs)*2625.5
                    backward_barrier = (e_ts-e_rhs)*2625.5
                    if not (forward_barrier < energy_cutoff or backward_barrier < energy_cutoff):
                        continue
                    if forward_barrier < best_barriers[0]:
                        best_barriers = (forward_barrier, best_barriers[1])
                    if backward_barrier < best_barriers[1]:
                        best_barriers = (best_barriers[0], backward_barrier)
                    if e_ts < best_ts:
                        best_ts = e_ts
                        best_spline = spline
                        best_step = step_id.string()
            if best_spline:
                template = ReactionTemplate.from_trajectory_spline(
                    best_spline,
                    barriers=best_barriers,
                    elementary_step_id=best_step
                )
                if template is not None:
                    self.add_template(template)

    def dump_svg_representation(self, path: str, index: Optional[int] = None) -> None:
        """Generate SVG representations of all templates.

        Parameters
        ----------
        path : str
            The path (folder) to generate the SVG files in.
        index : Optional[int], optional
            If given only generate an SVG for the template with the given index,
            by default None
        """
        def __dump_by_index(index: int):
            assert index < len(self.__unique_templates)
            svgs: List[svgutils.compose.SVG] = []
            fnames: List[str] = []
            for side in ['lhs', 'rhs']:
                count = 0
                for mol in self.__unique_templates[index].shelled_nuclei[side][0]:
                    for fragment in mol:
                        count += 1
                        fname = os.path.join(path, f'{side}-{count:03d}.svg')
                        fnames.append(fname)
                        write_molecule_to_svg(fname, fragment)
                        svgs.append(svgutils.compose.SVG(fname))
                if side == 'lhs':
                    middle = len(svgs)
            max_height = max(svgs, key=lambda x: x.height).height
            shifts = [0]
            for s in (svgs[0:-1]):
                shifts.append(shifts[-1]+s.width)
                if len(shifts) == middle+1:  # pylint: disable=possibly-used-before-assignment
                    shifts[-1] += 100
            for i, s in enumerate(svgs):
                s.move(shifts[i], (max_height-s.height)/2)
            svgutils.compose.Figure(
                sum([x.width for x in svgs])+100, max_height,
                *svgs
            ).save(os.path.join(path, f'template-{index:04d}.svg'))
            for fname in fnames:
                os.remove(fname)
        if index is not None:
            __dump_by_index(index)
        else:
            for i in range(len(self.__unique_templates)):
                __dump_by_index(i)
