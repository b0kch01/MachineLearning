# Loading and Saving Models
## About
Saving the trained model into a file can help save a lot of time
For example, your program might need a few hours to train. However, you can train the model once, and run tests on it multiple times without another long training session.

Dataset: https://github.com/zalandoresearch/fashion-mnist


# How to use?
## Using Pre-Trained Model
Just download the files, make sure the model is in same folder, and run:
```
python LoadModel.py
```
## Training your own model
Just donwload files except the pretrained model and run:
```
python SaveModel.py
```
Then, run:
```
python LoadModel.py
```
