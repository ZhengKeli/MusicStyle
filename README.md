# Music Style Recognition
This project is for music style recognition.
To run the model, you can follow the steps in this file.

## Python Environment
We do this project with Python 3.7.
The required packages are listed in file `requirements.txt`.

## Prediction
This project includes trained weights of the neural network.
Weights of baseline model are in folder `logs_fe`. Weights of our model are in folder `logs_sp`. In these folders there are also training logs, which can be opened with tensorboard.

You can directly run our model with trained weights by executing script `predict_sp.py`. But before that, you may need to changed the configurations inside this script (for example, to specify the input audio file).

## Dataset
We use GTZAN Genre Collection as our dataset for training, test and validation. This pack does no includes the dataset (because it is to big). If you want to validate or train the model, you need to download the dataset from [here](http://marsyas.info/downloads/datasets.html "GTZAN Genre Collection").

The dataset should be placed in folder `data`. That is, under folder `data` there should be folders of 10 genres (blues, classical, country, ..., rock).

## Data Splitting
To validate the model, we must know how the dataset was split.
We split the model into 3 subsets - training set, test set and validation set. 

Script `split.py` can be used to generate "ref files" of subsets. In such "ref files" there are paths of audio files in this subset. During the splitting, the audio files are shuffled.

This pack includes the "ref files" which were used by us for training. So do **NOT** execute `split.py` if you want to validate the model with out trained weights. Otherwise, the "ref files" will be overwritten. With different splitting, the result of validation will be not credible.

## Feature Extraction
Before validating and training the model we need to extract the features and spectrograms from audio files. 

Execute script `extract.py` to extract them. 

The extracted features and spectrograms will be stored as `*.npy` files next to the audio file. 

## Validation
You can execute script `valid_fe.py` to validate the baseline model, or execute script `valid_sp.py` to validate our model.

## Training
You can execute script `train_fe.py` to train the baseline model, or execute script `train_sp.py` to train our model.
