Changelog
=========

Release 2.0.1
-------------

- Add reference to SCINE review article to README

Release 2.0.0
-------------

- Allow None as energy cutoff
- Update address in license

Release 1.0.0
-------------

Initial Features:
 - Four reaction template flavors (many features are only geared towards the first two):
    - 'minimal' and 'minimal' with shape dressing
    - 'minimal_shell' (the minimal template type with nearest neighbors added for each reactive atom)
    - 'fragment' (all atom of one molecule have to form a single graph)
    - 'fragment_shell' (all atom of one molecule have to form a single graph, neighbors are added)
 - Deduplication of reaction templates based on a template graph representation
 - A database to hold reaction templates
 - Automated template extraction from SCINE databases
 - Template application to new molecules
 - Incomplete atom mapping features for arbitrary molecules
