# Loading and Saving Models
Saving the trained model into a file can help save alot of time
For example, your program might need a few hours to train. However, you can train the model once, and run tests on it multiple times without another long training session.

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
