# SpaceNet 6 - Building Segmentation
## CZ4042 Neural Networks and Deep Learning Project
### Installation Instructions
#### Cloning the Main Repository
In this repository, you will find our main source code with utility functions and driver code.
```bash
git clone https://github.com/reubenwong97/spacenet_6.git
conda env create -f environment.yml
conda activate tf2.1
```
Please clone the repository and create a new environment with the required cloud-hosted packages through the provided environment.yml file to ensure smooth running of code. Additionally, we have only tested this code on SCSE's GPU cluster, and it would be best to run any testing on the server.

#### Installing Customised Repository
Next, we need to install our customised repositories from source. Credits go to [qubvel](https://github.com/qubvel) for the base source code. We made changes to these repositories and thus, you will need to install ours.

**IMPORTANT**: Please clone and install this repository outside of the spacenet_6 folder to avoid python import issues. Also, please be sure to install `classification_models_dev` first as it is a dependency for `segmentation_models_dev`. 
```bash
# tf2.1 conda environment should be active when installing
git clone https://github.com/reubenwong97/models_dev.git 
cd classification_models_dev
pip install .
cd ../segmentation_models_dev
pip install .
```
#### Downloading Data and Placing them in Source Tree
Data can be downloaded [here](https://entuedu-my.sharepoint.com/:u:/g/personal/wong1109_e_ntu_edu_sg/EThP2bfs9ZtPq29YXvwQHN0B5wLWUHGGrd1fz8ax1Z0-0Q?e=za1iGJ).

Please extract the folder to obtain a `data_project` directory containing `train` and `test` directories. You should find 2 `.tfrecords` files inside `data_project/train` and 1 `.tfrecords` file inside `data_project/test`. Place this `data_project` directory inside the root directory of `spacenet_6`. 

To run scripts on SCSE’s GPU server, please run from the project root directory which has the job.sh provided. As an example, to run the architecture_trial for resnet18, run the following command from project root in spacenet_6:
```bash
sbatch job.sh scripts/architecture_trial/architecture_trial_resnet18.py 
```
For reference, refer to the directory structure below: 
```bash
├───archive
├───data_generation
├───data_project
│   ├───test # SN_6_test.tfrecords in this folder
│   │   ├───img
│   │   └───mask
│   └───train # SN_6.tfrecords and SN_6_val.tfrecords in this folder
│       ├───img
│       └───mask
├───results
│   ├───checkpoints
│   ├───figures
│   ├───histories
│   ├───predictions
│   ├───sample_figs
│   └───summary_figures
├───scripts
│   ├───architecture_trial
│   ├───data_augmentation
│   ├───external_paramater_optimizer
│   ├───external_parameter_decoderblocktype
│   ├───external_parameter_decoderusebatchnorm
│   ├───external_parameter_learningrate
│   ├───external_parameter_loss
│   ├───internal_parameter_activation
│   ├───internal_parameter_decoderdroprate
│   └───internal_parameter_decodernorm
├───utils
```

### SpaceNet 6
SpaceNet 6 Challenge dataset - just download train set (download through AWS CLI)
https://spacenet.ai/sn6-challenge/

SpaceNet 6 Github - Winners' solutions
https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions


### Packages
Fiona package - interact with geojson files
https://pypi.org/project/Fiona/

Rasterio package - interact with .tiff image files
https://pypi.org/project/rasterio/

Opencv - install through pip

Segmentation models
https://segmentation-models.readthedocs.io/en/latest/
