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

  If ```adversarial_training = True``` : Espilon and beta are used in the adversiarial loss. **Espilon** define the perturbations l2 norm that are added in the formula that generate the Adversarial Examples :


  <img alt="Formula e_adv" src="https://github.com/ClementPerroud/Adv-ALSTM/blob/main/readme_images/e_adv.JPG?raw=true" height = "30"/>


  <img alt="Formula r_adv" src="https://github.com/ClementPerroud/Adv-ALSTM/blob/main/readme_images/r_adv.JPG?raw=true" height = "30"/>

**Beta** is then use to weight the Adversarial loss generated with the Adversarial example following the formule bellow :

  <img alt="Formula general loss" src ="https://github.com/ClementPerroud/Adv-ALSTM/blob/main/readme_images/global_loss.JPG?raw=true" height = "55"/>



- **learning_rate** : float (optional, default : 1E-2)

  Define the learning rate used to initialized the Adam optimzer used for training.
- **dropout** : float (optional, default : 0.0)
- **l2** : float (optional, default : 1E-2)

  Define l2 regularization parameter
- **attention** : boolean (optional, default : True)

  If ```True```, the model will use the TemporalAttention after the LSTM layer to generate the Latent Space. If ```False```, the model will simply take the last hidden state of the LSTM (use ```return_sequences = False```)
- **hinge** : boolean (optional, default : True)
  If ```True```, the model will use the Hinge loss to perform training. If ```False```, the model use the Binary Cross-Entropy loss.
- **adversarial_training** : boolean (optional, default : True)

  If ```True```, the model will generate a Adversarial Loss from the Adversarial exemple that will be added to the global loss. If ```False```, the model will be training without adversarial example and loss.
- **random_perturbations** : boolean (optional, default : False)
  Define how the perturbations are created.
  If ```False``` (default), the perturbations are generated following the papier guidline with :

  <img alt="Formula g_s gradient" src = "https://github.com/ClementPerroud/Adv-ALSTM/blob/main/readme_images/g_s.JPG?raw=true" height = "30" />

  ```g``` is computed with ```tape.gradient(loss(y, y_pred), e)```

  If ```True```, the pertubations are randomly generated instead of being gradient oriented.

  ```g``` is computed with ```tf.random.normal(...)```