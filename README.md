**AutoCompress** provides scripts that use Bayesian Optimzer and Distiller to compress networks. This repository is  a fork of Distiller, with addition of few scripts in classifier compression example/

## Installation

These instructions will help get Distiller up and running on your local machine.
1. [Clone Autocompress](#clone-autocompress)
2. [Create a Python virtual environment](#create-a-python-virtual-environment)
3. [Install dependencies](#install-dependencies)

Notes:
- Distiller has only been tested on Ubuntu 16.04 LTS, and with Python 3.5.
- If you are not using a GPU, you might need to make small adjustments to the code.

### Clone Autocompress
Clone the Autocompress code repository from github:
```
$ git clone https://github.com/omernaeem/autocompress.git
```
The rest of the documentation that follows, assumes that you have cloned your repository to a directory called ```distiller```. <br>

### Create a Python virtual environment
We recommend using a [Python virtual environment](https://docs.python.org/3/library/venv.html#venv-def), but that of course, is up to you.
There's nothing special about using Distiller in a virtual environment, but we provide some instructions, for completeness.<br>
Before creating the virtual environment, make sure you are located in directory ```distiller```.  After creating the environment, you should see a directory called ```distiller/env```.
<br>
#### Using virtualenv
If you don't have virtualenv installed, you can find the installation instructions [here](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).

To create the environment, execute:
```
$ python3 -m virtualenv env
```
This creates a subdirectory named ```env``` where the python virtual environment is stored, and configures the current shell to use it as the default python environment.

#### Using venv
If you prefer to use ```venv```, then begin by installing it:
```
$ sudo apt-get install python3-venv
```
Then create the environment:
```
$ python3 -m venv env
```
As with virtualenv, this creates a directory called ```distiller/env```.<br>

#### Activate the environment
The environment activation and deactivation commands for ```venv``` and ```virtualenv``` are the same.<br>
**!NOTE: Make sure to activate the environment, before proceeding with the installation of the dependency packages:<br>**
```
$ source env/bin/activate
```

### Install the package
Finally, install the Distiller package and its dependencies using ```pip3```:
```
$ cd distiller
$ pip3 install -e .
```
This installs Distiller in "development mode", meaning any changes made in the code are reflected in the environment without re-running the install command (so no need to re-install after pulling changes from the Git repository).

PyTorch is included in the ```requirements.txt``` file, and will currently download PyTorch version 1.0.1 for CUDA 9.0.  This is the setup we've used for testing Distiller.

## Getting Started

You can jump head-first into some limited examples of network compression, to get a feeling for the library without too much investment on your part.  


### Example invocations

Various scripts for Resnet20, Plain20, Alexnet and Mobilenet are present in classifier compression directory

Most of the scripts take two arguments Theta(which assigns importance to compression vs accuracy) and number of iterations.

```
$ cd distiller/examples/classifier_compression
$ python3 MobileNetStage1.py <Theta> <Iterations>
```
## Built With

* [PyTorch](http://pytorch.org/) - The tensor and neural network framework used by Distiller.
* [Jupyter](http://jupyter.org/) - Notebook serving.
* [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) - Used to view training graphs.

