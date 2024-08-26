.. image:: docs/source/res/art_logo.png
   :alt: SCINE Art

.. inclusion-marker-do-not-remove

SCINE Art: Automated Reaction Templates
=======================================

Introduction
------------

With SCINE Art you can extract reaction templates from complex chemical reaction
networks in an automated fashion. Based on this Python framework, workflows can
be built that apply the extracted templates to new network explorations.

License and Copyright Information
---------------------------------

Art is distributed under the BSD 3-clause "New" or "Revised" License.
For more license and copyright information, see the file ``LICENSE.txt`` in the
repository.

Installation
------------

Prerequisites
.............

The key requirements for Art are the Python packages ``scine_utilities``
and ``scine_molassember``. Optionally the ``scine_database`` can be used.
These packages are available from PyPI and can be installed using ``pip``.
However, these packages can also be compiled by hand. For the latter case please
visit the repositories of each of the packages and follow their guidelines.

Installation
............

Art can be installed using pip (pip3) once the repository has been cloned:

.. code-block:: bash

   git clone <art-repo>
   pip install ./art

A non super user can install the package using a virtual environment, or
the ``--user`` flag.

The documentation can be found online, or it can be built using:

.. code-block:: bash

   cd art
   make -C docs html

It is then available at:

.. code-block:: bash

   <browser name> docs/build/html/index.html

In order to build the documentation, you need a few extra Python packages wich
are not installed automatically together with Art. In order to install them,
run

.. code-block:: bash

   cd art
   pip install -r requirements-dev.txt

How to Cite
-----------

When publishing results obtained with Art, please cite the corresponding
release as archived on `Zenodo <https://zenodo.org/doi/10.5281/zenodo.7928113>`_
(please use the DOI of the respective release).

In addition, we kindly request you to cite the following article when using Art:

J. P. Unsleber, "Accelerating Reaction Network Explorations with Automated Reaction
Template Extraction and Application", *J. Chem. Inf. Model.*, **2023**, *63*, 3392.

Furthermore, when publishing results obtained with any SCINE module, please cite the following paper:

T. Weymuth, J. P. Unsleber, P. L. Türtscher, M. Steiner, J.-G. Sobez, C. H. Müller, M. Mörchen,
V. Klasovita, S. A. Grimmel, M. Eckhoff, K.-S. Csizi, F. Bosia, M. Bensberg, M. Reiher,
"SCINE—Software for chemical interaction networks", *J. Chem. Phys.*, **2024**, *160*, 222501
(DOI `10.1063/5.0206974 <https://doi.org/10.1063/5.0206974>`_).

Support and Contact
-------------------

In case you should encounter problems or bugs, please write a short message
to scine@phys.chem.ethz.ch.
