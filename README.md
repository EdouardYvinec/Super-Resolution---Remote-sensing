# Super-Resolution---Remote-sensing
Super Résolution - Remote Sensing

This the git repo of our work on the question : "Can deep-learning-based single-image super-resolution be invariant
to subpixel translations?"
The python code is located in the src folder. It contains both tensorflow 2.0 and tensorflow 1.14 code, therefore, some tensorflow warnings may pop up when usage. Everything is run from the src folder via the main.py file.

# Setting Up : test the models
If you only want to run the tests everything is provided in this repo. Simply run the main.py

# Setting Up : train the models
If you want to retrain the models, you should first add the training set from https://data.vision.ee.ethz.ch/cvl/DIV2K/ (the 4 times down sampled (bi-cubic) data) in the dataset folder.
After this, delete all the files in checkpoint_model and run the main.py file.
Our code will check for those files and train if and only if they aren't provided (which they are by default).

# Contact :
If you have any questions/issues please contact us at 
  ey@datakalab.com
  durand.jeffery@gmail.com
