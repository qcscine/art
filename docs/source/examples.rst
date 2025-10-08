
Examples
========

Here we provide a minimalistic example of how to use ``scine_art`` to generate reaction templates from a previously constructed reaction network.

.. code-block:: python

      from scine_art.database import ReactionTemplateDatabase
      
      rtd = ReactionTemplateDatabase() # Call the class

      args = {
      "host": "localhost",             # where is the MongoDB hosted
      "port": 27017,                   # which port does it use
      "name": "default",               # which name does the MongoDB database have
      "energy_cutoff": 250.0,          # which energy threshold should be used to disregard reactions
      "model": None,                   # in case the collection contains different models
      "gibbs_free_energy": False,      # electronic or Gibbs energies
      }

      rtd.add_scine_database(**args)   # adding the database will already trigger the calculation of the templates

      print("There are the following number of template reactions:", rtd.template_count())
      
      path_svg = '/home/username/'
      rtd.dump_svg_representation(path_svg) # saves SVG images of the templates of the network

      path_pickle = '/home/username/template.pkl'
      rtd.save(path_pickle) # saves the template object so that it can be loaded in an exploration


