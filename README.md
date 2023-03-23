# Adv-ALSTM
An attempt to replicate Adv-ASLTM model with Tensorflow 2 from papier :

@article{feng2019enhancing,
  title={Enhancing Stock Movement Prediction with Adversarial Training},
  author={Feng, Fuli and Chen, Huimin and He, Xiangnan and Ding, Ji and Sun, Maosong and Chua, Tat-Seng},
  journal={IJCAI},
  year={2019}
}

This model can be use for Binary Classification tasks and will be used bellow as a stock movement classifier (UP/DOWN).

## Table of content
- Getting started
  - Description
  - Dependencies
- How to use the Adv-ASLTM model?
  - Installation
  - Use

## Getting started
### Description
The projet is all about reproducting a working version of Adversarial Attention Based LSTM for Tensorflow 2. This new version is available in ```AdvALSTM.py``` with the object ```AdvLSTM```.
More details of my version bellow.

I finally updated the author's code for it to run with Tensorflow 2.x and compare my results with his. The updated code is available in the folder ```original_code_updated``` .
```
│   .gitignore
│   README.md
│   AdvALSTM.py
│   preprocessing.py
│   replicate_result.py
├───data
│   └───stocknet-dataset
│       └───price
│           │   trading_dates.csv
│           ├───ourpped/
│           └───raw/
└───original_code_updated
        evaluator.py
        load.py
        pred_lstm.py
        __init__.py


```
### Dependancies
- Tensorflow : 2.9.2

## How to use the Adv-ASLTM model?
### Installation
Download the ```AdvASLTM.py``` file and place it in your project folder.

```python
from AdvLSTM import AdvLSTM
```
### Use
To create a AdvLSTM model, use :

```python
model = AdvLSTM(
  units, 
  epsilon, 
  beta, 
  learning_rate = 1E-2, 
  dropout = None, 
  l2 = None, 
  attention = True, 
  hinge = True, 
  adversarial_training = True, 
  random_perturbations = False)
```
The AdvLSTM object is a subclass of ```tf.keras.Model```. So you can easily train it as you would normaly do with a Tensorflow 2 model : 
```python
model.fit(
  X_train, y_train, 
  validation_data = (X_validation, y_validation),
  epochs = 200, 
  batch_size = 1024
  )
```
The model only accepts:
-  **y** : labelled as binary classes (0 or 1), even when using Hinge loss !

```(nb_sequences, )```
-  **x** : sequences of lenght T, with n features.

```(nb_sequences, T, n)```

### Documentation
```python
class AdvALSTM.AdvALSTM(**params):
__init__(self, units, epsilon = 1E-3, beta =  5E-2, learning_rate = 1E-2, dropout = None, l2 = None, attention = True, hinge = True, adversarial_training = True, random_perturbations = False)
```
- **units** : int (required)

  Specify the number of units of the layers (Dense, LSTM and Temporal Attention) contained in the model.

- **epsilon** : float (optional, default : 1E-3)
- **beta** : float (optional, default : 5E-2)

If ```adversarial_training = True``` : Espilon and beta are used in the adversiarial loss. Espilon define the perturbations l2 norm that are added in the formule
[alt text](https://github.com/ClementPerroud/Adv-ALSTM/readme_images/e_adv.jpg?raw=true)

- **learning_rate** : float (optional, default : 1E-2)
- **dropout** : float (optional, default : 0.0)
- **l2** : float (optional, default : 1E-2)
- **attention** : boolean (optional, default : True)
- **hinge** : boolean (optional, default : True)
- **adversarial_training** : boolean (optional, default : True)
- **random_perturbations** : boolean (optional, default : False)