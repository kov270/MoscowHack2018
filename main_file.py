from __future__ import print_function

import sys
import os
import time
import glob

import numpy as np
import csv
import pandas as pd
#import theano
#import theano.tensor as T

import lasagne

def load():
	data = []
	s = os.path.dirname(os.path.abspath(__file__))+"/Bosphorus2D/"
	for fe in os.listdir(os.getcwd()+ "/Bosphorus2D/"):
		l = os.getcwd()+ "/Bosphorus2D/"
		for filename in glob.glob(str(l+'/'+fe+'/')+'*.txt'):
			b = []
			f = open(filename,'r')
			t = f.readlines()
			emotion = t[1]
			lines = t[2:]
			for l in lines:
				buf = l.rstrip().split(" ")
				buf_name = buf[:-2]
				#print(buf)
				x = buf[-2]
				y = buf[-1]
				b.append([" ".join(str(x) for x in buf_name),x,y])
			data.append([emotion.rstrip(),b])
			break
	return data

d = load()

raw_data = {'Outer left eyebrow_x':[], 'Outer left eyebrow_y':[], 'Middle left eyebrow_x':[], 'Middle left eyebrow_y':[], 'Inner left eyebrow_x':[], 'Inner left eyebrow_y':[], 'Inner right eyebrow_x':[], 'Inner right eyebrow_y':[], 'Middle right eyebrow_x':[], 'Middle right eyebrow_y':[], 'Outer right eyebrow_x':[], 'Outer right eyebrow_y':[], 'Outer left eye corner_x':[], 'Outer left eye corner_y':[], 'Inner left eye corner_x':[], 'Inner left eye corner_y':[], 'Inner right eye corner_x':[], 'Inner right eye corner_y':[], 'Outer right eye corner_x':[], 'Outer right eye corner_y':[], 'Nose saddle left_x':[], 'Nose saddle left_y':[], 'Nose saddle right_x':[], 'Nose saddle right_y':[], 'Left nose peak_x':[], 'Left nose peak_y':[], 'Nose tip_x':[], 'Nose tip_y':[], 'Right nose peak_x':[], 'Right nose peak_y':[], 'Left mouth corner_x':[], 'Left mouth corner_y':[], 'Upper lip outer middle_x':[], 'Upper lip outer middle_y':[], 'Right mouth corner_x':[], 'Right mouth corner_y':[], 'Upper lip inner middle_x':[], 'Upper lip inner middle_y':[], 'Lower lip inner middle_x':[], 'Lower lip inner middle_y':[], 'Lower lip outer middle_x':[], 'Lower lip outer middle_y':[], 'Chin middle_x':[], 'Chin middle_y':[], 'Left temple_x':[], 'Left temple_y':[], 'Right temple_x':[], 'Right temple_y':[],'Emotion':[]}

p_ar = ['Outer left eyebrow_x', 'Outer left eyebrow_y', 'Middle left eyebrow_x', 'Middle left eyebrow_y', 'Inner left eyebrow_x', 'Inner left eyebrow_y', 'Inner right eyebrow_x', 'Inner right eyebrow_y', 'Middle right eyebrow_x', 'Middle right eyebrow_y', 'Outer right eyebrow_x', 'Outer right eyebrow_y', 'Outer left eye corner_x', 'Outer left eye corner_y', 'Inner left eye corner_x', 'Inner left eye corner_y', 'Inner right eye corner_x', 'Inner right eye corner_y', 'Outer right eye corner_x', 'Outer right eye corner_y', 'Nose saddle left_x', 'Nose saddle left_y', 'Nose saddle right_x', 'Nose saddle right_y', 'Left nose peak_x', 'Left nose peak_y', 'Nose tip_x', 'Nose tip_y', 'Right nose peak_x', 'Right nose peak_y', 'Left mouth corner_x', 'Left mouth corner_y', 'Upper lip outer middle_x', 'Upper lip outer middle_y', 'Right mouth corner_x', 'Right mouth corner_y', 'Upper lip inner middle_x', 'Upper lip inner middle_y', 'Lower lip inner middle_x', 'Lower lip inner middle_y', 'Lower lip outer middle_x', 'Lower lip outer middle_y', 'Chin middle_x', 'Chin middle_y', 'Left temple_x', 'Left temple_y', 'Right temple_x', 'Right temple_y','Emotion']

print(len(p_ar))

for e in d:
	for t in e[1]:
		try:
			if t[0]+'_x' == "Publishable_x":
				break
			else:
				raw_data[t[0]+'_x'].append(t[1])
				raw_data[t[0]+'_y'].append(t[2])
		except KeyError:
			#print(t[0])
			pass
	raw_data['Emotion'].append(e[0])

df = pd.DataFrame.from_dict(raw_data, orient='index').transpose()

#df.transpose()
df.to_csv('test.csv')
