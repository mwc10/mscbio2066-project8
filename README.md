# Assignment 8 Code

Simple XGBoost based model for a retroactive entry in the [IDG-DREAM Drug-Kinase Binding Prediction Challenge](https://www.synapse.org/#!Synapse:syn15667962/wiki/583663).

## Files

The final model is in `model.py`. It can be run as a standalone script, or imported as a python module. It was trained by providing a kinase index and data to `train-model.py`, which produces a collection of `.ubj` XGBoost settings and `.csv` of training performance. Those files can be packaged into one model `.tar.gz` with `package-model.py`. 

A set of pretrained models are provided in the `pretrained` directory, along with a processed dataset that can be used to train more models.
