# EDRAM
Enriched Deep Recurrent Visual Attention Model
-----------------------------------------------
The repository contains implementation of Enriched Deep Recurrent Visual Attention Model for MNIST Cluttered (https://github.com/deepmind/mnist-cluttered). The original paper in proceeding to the conference.


Dependencies
------------
 * [Blocks](https://github.com/bartvm/blocks) follow
the [install instructions](http://blocks.readthedocs.org/en/latest/setup.html).
This will install all the other dependencies for you (Theano, Fuel, etc.).
 * [MNIST Cluttered](https://github.com/deepmind/mnist-cluttered)
 * [Bokeh](http://bokeh.pydata.org/en/latest/docs/installation.html) 0.8.1+

Dataset
----

To be able to train the model you need to build MNSIT Cluttered dataset by running script from the fork (https://github.com/ablavatski/mnist-cluttered)

	luajit mnist_cluttered_gen.lua
	python mnist_cluttered.py --path

and setup the location of your data directory:

    export FUEL_DATA_PATH=/home/user/data
	
Training
-----------------------

To train the model with a basic setup you need to run the script
	
	cd edram
	python train_mnist_cluttered.py
	
Usually it takes around 60 epoch to train the model. After training in the folder you can find
 
 * a pickle of the best model
 * a pickle of the log
 * the txt log file of the training