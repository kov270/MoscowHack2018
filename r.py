import os
import lasagne
import theano
import numpy as np
import cPickle as pickle
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
 
from matplotlib import pyplot
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.init import Uniform
from nolearn.lasagne import NeuralNet
from lasagne.layers import *
from theano.tensor.nnet import sigmoid
 
 
 
 
class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2**(1.0/self.half_life)
        self.variable.set_value(np.float32(self.target + delta))
 
def float32(k):
    return np.cast['float32'](k)
 
 
def load(test=False, cols=None):
	FTRAIN = "/Users/Documents/train.csv"
	FTEST = "/Users/Documents/test.csv"
	fname = FTEST if test else FTRAIN

	df = read_csv(os.path.expanduser(fname))

	df = df.dropna()  

	if not test:
	    bl = []    
	    for e in df['Emotion'].values:

	        l = []
	        for i in range(0,53):
	            if e-1 != i:
	                l.append(0)
	            else:
	                l.append(1)
	        bl.append(l)
	    X = np.vstack(bl)

	    X = X.astype(np.float32)
	    df = df.drop('Emotion', 1)
	else:
	    X = None


	df = df.drop('Unnamed: 0', 1)
	
	y = df.values
	y = y.astype(np.float32)
	#X, y = shuffle(X, y, random_state=42)  # shuffle train data

	return y, X
def train():
    net = NeuralNet(
            layers=[  
                ('input', InputLayer),
                ('dropout0', DropoutLayer),
                ('hidden1', DenseLayer),
                ('hidden2', DenseLayer),
                ('output', DenseLayer),
                ],
 
            input_shape=(None,48),
            dropout0_p=0.05,
            hidden1_num_units=2000,
            hidden1_W=Uniform(),
            hidden2_num_units=2000,
            hidden2_W=Uniform(),
 
            output_nonlinearity=sigmoid,
            output_num_units=53,
            update=nesterov_momentum,
            update_learning_rate=theano.shared(np.float32(0.001)),
            update_momentum=theano.shared(np.float32(0.9)),  
 
           
            # Decay the learning rate
            on_epoch_finished=[
                    AdjustVariable(theano.shared(float32(0.03)), target=0, half_life=4),
                    ],
 
            # This is silly, but we don't want a stratified K-Fold here
            # To compensate we need to pass in the y_tensor_type and the loss.
            regression=True,
            #y_tensor_type = T.imatrix,
            #objective_loss_function = binary_crossentropy,
           
            #regression=True,
            max_epochs=20000,
            verbose=1,
            )
    X, y = load()
 
    net.fit(X, y)
 
    with open('net3.pickle', 'wb') as f:
        pickle.dump(net, f, -1)
 
 
def use():
    dic = pickle.load(open( "/Users/Documents/dump_of_emotion_by_index.p", "rb" ))
    with open('net3.pickle', 'rb') as f:  # !
        net = pickle.load(f)  # !
    X, _ = load(test=True)
    y_pred = net.predict(X)
    for e in y_pred:
        r = e.tolist()
        for y in r:
            print(dic.get(str(r.index(y)+1)),r.index(y)+1,y)
        print("Max r:",max(r))
    
 
 
#train()
 
 
use()
