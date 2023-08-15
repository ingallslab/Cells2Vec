# Cells2Vec: Bridging The Gap Between Simulations And Experiments Using Causal Representation Learning -- Code

This is the code corresponding to the experiments conducted for the work "Cells2Vec" where we used Causal Representation Learning to learn useful representations from high dimensional simulations and used Experiments to validate these representations, as well as estimate input parameters using Regression.

## Requirements

Experiments were done with the following package versions for Python 3.10:
 - Numpy (`numpy`) v1.21.5;
 - Matplotlib (`matplotlib`) v3.5.3;
 
 - Pandas (`pandas`) v2.0.1;
 - `cellmodeller-ingallslab` vx.x.x( To read raw simulations into torch);
 - PyTorch (`torch`) v1.13.0 with CUDA 11.0;
 - Scikit-learn (`sklearn`) v1.3.0;
 - XGBoost (`xgboost`) v1.7.5.

This code should execute correctly with updated versions of these packages. Use the `requirments.txt` file to install these, except the `cellmodeller-ingallslab` package.

## Datasets

We used `1000` simulations by sampling parameters `Gamma`, `Reg_Param` and `Adhesion` (Ref: CellModeller documentation).
We had `100` parameter sets and `10` simulations for each set.

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

### Results and Visualization

 - `checkpoints` directory: Plots and CSV results of regression will be generated here.
 - `runs` directory: Tensorboard logs saved here

## Usage

### Training using default hyperparemeters

To train a model on the Mallat dataset from the UCR archive:

`python3 ucr.py --dataset Mallat --path path/to/Mallat/folder/ --save_path /path/to/save/models --hyper default_hyperparameters.json [--cuda --gpu 0]`

Adding the `--load` option allows to load a model from the specified save path.
Training on the UEA archive with `uea.py` is done in a similar way.

### Further Documentation

See the code documentation for more details. `ucr.py`, `uea.py`,
`transfer_ucr.py`, `combine_ucr.py` and `combine_uea.py` can be called with the
`-h` option for additional help.

### Hyperparameters

Hyperparameters are described in Section S2.2 of the paper.

For the UCR and UEA hyperparameters, two values were switched by mistake.
One should read, as reflected in [the example configuration file](default_hyperparameters.json):
> - number of output channels of the causal network (before max pooling): 160;
> - dimension of the representations: 320.
>
instead of
> - number of output channels of the causal network (before max pooling): 320;
> - dimension of the representations: 160.

## Pretrained Models

Pretrained models are downloadable at [https://data.lip6.fr/usrlts/](https://data.lip6.fr/usrlts/).
