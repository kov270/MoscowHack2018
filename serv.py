# test_server.py
import webob
from paste import httpserver
from PIL import Image
import glob, os
import cgi
import cPickle as pickle
import numpy as np
import math as m
import pandas as pd

from StringIO import StringIO
from matplotlib import pyplot
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
 
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

dic = pickle.load(open( "/Users/Konstantin/Documents/kyrsovaja/dump_of_emotion_by_index.p", "rb" ))
with open('net3.pickle', 'rb') as f:  # !
        net = pickle.load(f)  # !


def load(test=False,fname = None):
 
    df = pd.DataFrame.from_csv(StringIO(fname), sep=",", parse_dates=False)
    df = df.dropna()  
    #df = df.drop('Emotion', 1)
    #df = df.drop('Unnamed: 0', 1)
    X = None
    y = df.values
    y = y.astype(np.float32)
    return y, X

 
def app(environ, start_response):

	request = webob.Request(environ)

	start_response("200 OK", [("Content-Type", "text/plain")])

	X, _ = load(test=True,fname = request.POST["file"].value)
	y_pred = net.predict(X)
	u = []
	for e in y_pred:
		r = e.tolist()
		for y in r:
			u.append([dic.get(str(r.index(y)+1)),y])
             #print()
        u.append(["Max r:",max(r)])
         #print("Max r:",max(r))
	#print(u)
    
	yield u
 
httpserver.serve(app, host='172.19.32.112', port=5001)