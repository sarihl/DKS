Hello!
This README is a quick guide to get you started with this deep learning project skeleton.

Goal:
In my experience, large Deep learning projects tend to have a lot of boilerplate code that is repeated over and over again.
This boilerplate code is often not very readable and can be hard to maintain, and adds friction to the development process as the project grows larger.
This project skeleton is an attempt to reduce the amount of boilerplate code that is needed to get started with a new project, and to make experimenting with new ideas easier and faster.
I hope that you might find this project useful!

Main components:
- a training loop based on PyTorch Lightning.
- a configuration system based on Hydra.
- a logging system based on Wandb.
- a component loading\initialization system based on fvcore registry.

Pipeline overview:
1. the configuration is loaded from a yaml file residing in ./options/ using Hydra.
2. a wandb run is created and the configuration is logged.
3. a model is created using the configuration.
4. a data module is created using the configuration.
5. the model is trained using the data module and the configuration.

---------------------
How to use:
In order to implement your project using this skeleton, you must implement the following components:
1. a model.
2. a data module.
3. a configuration file.
4. (optional) a network architecture.

** 1 Creating a model **
- create a new file named X_model.py (X is whatever) in ./models/ and implement your model code there.
- your model must receive a Dictionary as the argument to its constructor.
- register your model using the @MODEL_REGISTRY.register() decorator.

** 2 Creating a data module **
TODO: write this!

** 3 Creating a configuration file **
- create a new file named X.yaml (X is whatever) in ./options/ and implement your configuration there.
- the configuration must be a valid Hydra configuration, that contains the entries in the skeleton configuration files.
# configuration details at the bottom of this file.

** 4 (optional) Creating a network architecture **
- create a new file named X_network.py (X is whatever) in ./networks/ and implement your architecture code there.
- register your architecture using the @NETWORK_REGISTRY.register() decorator.

---------------------
Configuration Details:
TODO: write this!




ENJOY!

