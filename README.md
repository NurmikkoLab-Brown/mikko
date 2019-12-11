[![DOI](https://zenodo.org/badge/210954471.svg)](https://zenodo.org/badge/latestdoi/210954471)

# mikko Neural Processing Toolkit

Mikko ("meeko") is a neural processing toolkit designed for large-scale computational neuroscience experiments. This 
software is developed and maintained by the [Nurmikko Lab](http://nurmikko.engin.brown.edu/) at Brown University.

Mikko is currently under development. Please check back regularly for new modules and updated documentation/examples. 

## Table of Contents

1. [Installation](#Installation)

2. [Getting Started](#GettingStarted)

3. [Documentation](#Documentation)

4. [Citing mikko](#Citing)

<a name="Installation"></a>
## 1. Installation

Mikko modules can be run as stand-alone containers or with Dockex.

To run modules stand-alone, install [Docker-CE](https://docs.docker.com/install/). For GPU support, you must also 
install the appropriate NVIDIA drivers and install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

To run modules with Dockex, install [Dockex](https://github.com/ConnexonSystems/Dockex). For GPU support, you must set 
the ```enable_gpus``` flag to ```true``` in the dockex configuration.

Each module runs as a self-contained container. No additional installation is necessary.

<a name="GettingStarted"></a>
## 2. Getting Started

### Stand-alone

When running modules stand-alone, you must handle the following manually:

* build the module image

* define a configuration JSON file that defines parameters, input pathnames, and output pathnames

* create a data directory for passing input files and storing output files

* run the module as a docker container with a volume linking to the data directory 

To build stand-alone images, point to the module's Dockerfile. For example, 

```bash
cd /path/to/mikko
docker build --tag lstm_rnn_image -f modules/decoders/keras/lstm_rnn/Dockerfile .
```

The module can then be run by first defining a run configuration JSON file. See ```test.json``` files in each module 
directory for examples.

When running stand-alone, you must also create a directory for storing the input and output data (e.g. ```/tmp/data```).

Then run the module. The JSON filename should be passed as an argument to the module. For example, 

```bash
docker run -it --rm -v /tmp/data:/tmp/data lstm_rnn_image test.json
```

Note that the JSON file must be accessible from within the docker container. This can be done by building the
file into the image or attaching a volume containing the file to the container. 

### Dockex

To run modules with Dockex, create an experiment file (see ```experiments/basic_auditory_experiment.py```), and copy 
your input data to the tmp_dockex_path (e.g. ```/tmp/dockex/data```).

Launch the experiment through the ```LAUNCH``` screen of the Dockex GUI. For example,

* Project Path: ```/path/to/mikko```
* Experiment Path: ```experiments/basic_auditory_experiment.py```

Set the number of desired CPU/GPU credits using the ```MACHINES``` screen. The ```PROGRESS``` screen will indicate 
when the experiment is complete.

Results will be saved to the tmp_dockex_path.

<a name="Documentation"></a>
## 2. Documentation

Documentation can be found [here](https://NurmikkoLab-Brown.github.io/mikko/).

<a name="Citing"></a>
## 4. Citing mikko

If you make use of this software for your work, please consider referencing the following:

### mikko
    @software{mikko,
      author       = {ChrisHeelan},
      title        = {NurmikkoLab-Brown/mikko: Initial Release},
      month        = nov,
      year         = 2019,
      publisher    = {Zenodo},
      version      = {v0.1.1},
      doi          = {10.5281/zenodo.3525273},
      url          = {https://doi.org/10.5281/zenodo.3525273}
    }
