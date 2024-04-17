# Projet de 2eme Annee ENSICAEN
**Cecile LU, Paul NGUYEN, Zeyd BOUMAHDI, Anis AHMED-ZAID, Noura OUTLIOUA**

contact : paul.nguyen@ecole.ensicaen.fr

Our project involves performing prediction tasks on encrypted data. To do this, we need to teach our model to recognize numerical images.

This branch contains :
- cnn_cats_dogs.py : loads dataset, creates and trains our model for later encryption with helayers
- encrypt_cats_dogs.py : loads and encrypts model, loads and encrypts test images, perform encrypted predictions
- activations.py : layers for our neural network
- utils.py : from IBM tutorial on helayers

## Dataset

Dataset used :
```
https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
```

Prerequisite : 
having a training_set and test_set both with subdirectories 'cats' and 'dogs'

## IBM fhe toolkit
```bash
git clone https://github.com/IBM/fhe-toolkit-linux.git
cd fhe-toolkit-linux/
sudo ./StartHELayers.sh python
sudo docker run -p 8080:8888 -d --rm --name helayers-pylab-s390x icr.io/helayers/helayers-pylab:latest
```
url : 
```http://localhost:8080/lab/?token=demo-experience-with-fhe-and-python```

## Results
### CNN with plain pictures
- training : 125s per epoch 
(batch_size = 100, 80 steps/epoch)

- precision : 0.9968

- prediction duration [75 - 85 ms] 

details :
- after 15 epoch : val_accuracy 0.733



### CNN + HElayers

#### cnn_cats_dogs.py
this program creates a model and trains it, using fhe encryption limits our model capacities because fhe does notsupport MaxPooling2D and Dropout layers

```bash
RuntimeError: Neural network architecture initialization from JSON failed: Neural network initialization from JSON encountered an operator type that is currently not supported: MaxPooling2D
```

this is because fhe only supports additions and products but not max(), this limits greatly our model's performances

first prediction :
```bash
Misc. initalizations
HE context initialized
loaded plain model
Profile ready. Batch size= 16
Encrypted network ready
Loaded test data
encryted test data
Predicted
Duration of predict: 7.858 (s)
Duration of predict per sample: 0.491 (s) (400 ms)
predictions [[-459.98687074  563.33570614]
 [-170.02127961  197.46265763]
 [-429.84242377  529.53916487]
 [-409.437333    476.33278837]
 [-441.84242066  476.89574164]
 [-112.54731279  137.81501224]
 [-472.32544444  564.26763841]
 [-842.49245486  973.06120955]
 [-139.09326657  159.20277754]
 [-731.86644046  905.04127645]
 [-194.97481495  234.3519231 ]
 [-279.8158711   323.93255546]
 [-661.36316843  764.91163111]
 [-467.23641055  568.34619144]
 [-333.9864512   354.00091547]
 [-496.42805962  591.12069488]]

Confusion Matrix:
[[ 0 16]
 [ 0  0]]
```

#### Better model
```python
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=3, padding='valid', input_shape=input_shape))
model.add(Flatten())
model.add(SquareActivation())
model.add(Dense(100))
model.add(SquareActivation())
model.add(Dense(num_classes, activation='softmax'))
```

#### results
After training our better performing model, we get better results

Recognizing cats and dogs :

batch size = 4
```bash
Duration of predict: 1.874 (s)
Duration of predict per sample: 0.469 (s)
predictions [[ -7.17411971   6.43331551]
 [ 17.68139776 -18.09483621]
 [  9.78133427 -10.5797041 ]
 [ 26.73074126 -26.44157767]]

Confusion Matrix:
[[3 1]
 [0 0]]
```

batch size = 8 
```bash
Duration of predict: 2.607 (s)
Duration of predict per sample: 0.326 (s)
predictions [[ -7.17411972   6.43331551]
 [ 17.68139774 -18.0948362 ]
 [  9.78133427 -10.5797041 ]
 [ 26.73074125 -26.44157767]
 [  8.22979703  -9.15138994]
 [ 11.98273023 -12.25144162]
 [ 28.56780586 -28.3615948 ]
 [  9.54165041  -9.42402878]]

Confusion Matrix:
[[7 1]
 [0 0]]
```
