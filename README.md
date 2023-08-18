# Cells2Vec: Bridging The Gap Between Simulations And Experiments Using Causal Representation Learning -- Code

This is the code corresponding to the experiments conducted for the work "Cells2Vec" where we used Causal Representation Learning to learn useful representations from high dimensional simulations and used Experiments to validate these representations, as well as estimate input parameters using Regression.

## Requirements

Experiments were done with the following package versions for Python 3.10:
 - Numpy (`numpy`) v1.21.5;
 - Matplotlib (`matplotlib`) v3.5.3;
 
 - Pandas (`pandas`) v2.0.1;
 - CellModeller (`cellmodeller-ingallslab`) vx.x.x( To read raw simulations into torch);
 - PyTorch (`torch`) v1.13.0 with CUDA 11.0;
 - Scikit-learn (`sklearn`) v1.3.0;
 - XGBoost (`xgboost`) v1.7.5.

This code should execute correctly with updated versions of these packages. Use the `requirments.txt` file to install these, except the `cellmodeller-ingallslab` package. \

Below is one of many ways to setup a Virtual environment: After cloning the repo, using a terminal session 


`mkdir env` \
 `cd env`\
 `virtualenv .` \
   `cd ..` \
`source env/bin/activate` \
 `pip install -r requirements.txt` 

## Datasets

We used `1000` simulations by sampling parameters `Gamma`, `Reg_Param` and `Adhesion` (Ref: CellModeller documentation) [here](https://github.com/cellmodeller/CellModeller/wiki).
We had `100` parameter sets and `10` simulations for each set.
Our dataset is available for download [here](https://drive.google.com/file/d/1WnxSY_DN2_Z3bSsZDkjnC-fngTstjFFO/view?usp=sharing), move it to the `Data` directory after downloading.

## Files

### Core

 - `losses.py` file: implements the triplet loss with custom distance functions, as well as regularization for the same;
 - `networks.py` file: implements encoder and its building blocks (dilated convolutions, causal CNN) as well as LSTM and GRU encoders;
 - `data_utils.py` file: implements custom PyTorch datasets, and code to sample triplets iteratively, and unravel a padded tensor, and read a raw simulation into a tensor;
 - `configs/configs.yaml` file: example of a YAML file containing the hyperparameters of a complete run of the code;
 - `eval.py` file: Code to calculate K-Means clustering metrics, and fit a XGBoost Regression model on embeddings of a trained encoder to estimate parameters;
 - `trainer.py` file: Code for model training
 - `visualize.py` file: Code for generating KMeans+PCA and TSNE Plots.
 - `main.py` file: Wrapper file for an end to end run.
 - `sim2data.py` file: Reads raw simulations into torch tensors. For `n` iterations (directories) of `k` simulations, groups the first file from all directories together, then the second ...`k`

### Results and Visualization

 - `checkpoints` directory: Plots and CSV results of regression will be generated here, along with the model checkpoint(s).
 - `runs` directory: Tensorboard logs saved here

## Usage

### Training using default hyperparemeters (Save your dataset as dataset.pt in the Data directory)

To train a model using default hyperparameters and to evaluate:

`python3 main.py --config configs/configs.yaml --selected_config default`


### Further Documentation

See the code documentation for more details. `main.py` can be called with the
`-h` option for additional help.

### Hyperparameters

Hyperparameters for the Encoder can be found [here](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/tree/master).

Hyperparameters for training:
 - `num_samples`: Total size of training dataset. Random triplets will be sampled to output the training dataset;
 - `num_val` : Number of classes to exclude from the training process (To test model's ability to generalize);
 - `val_indices`: Manually select classes to exclude from training (Random if not specified).
 - `split_idx`: (See `data_utils.py`) Splits the training set into a training and validation (early stopping condition) sets.


## Pretrained Models

Coming soon
