# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:03:04 2019

@author: Navein Kumar
"""

import pandas as pd
import numpy as np

import tflearn
import tensorflow
import random


df = pd.read_csv('s.csv')

inp =df.to_numpy()

final_inp =[]
final_inp=inp[:,0:4]
output=inp[:,4]

final_output=[]

for i in output:
    if i=="Cancer":
        final_output.append([1,0,0,0])
        
    if i=="Diabeties":
        final_output.append([0,1,0,0])
        
        
    if i=="Stomach":
        final_output.append([0,0,1,0])
        
        
        
    if i=="Heart":
        final_output.append([0,0,0,1])  
        
        
        
final_output = np.array(final_output)        
    
        
        
        
        


rbc=np.array(df['RBC'])
sugar=np.array(df['Sugar'])
acid=np.array(df['Acidity'])
coles=np.array(df['Colestrol'])

output=np.array(df['Disease'])


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, 4])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 4, activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(final_inp, final_output, n_epoch=250, batch_size=1, show_metric=True)
    model.save("model.tflearn")


results = model.predict([[40,40,40,47]])
results_index = np.argmax(results)

food={0:[["maidha","Coliflower","msdklsfngskl"]],1:[["maidha","Coliflower","msdklsfngskl"]],2:[["maidha","Coliflower","msdklsfngskl"]],3:[["maidha","Coliflower","msdklsfngskl"]]}  

if results_index==0:
    print("Cancer")
    print(food[0])
        
if results_index==1:
    print("Diabeties")
    print(food[1])
        
        
if results_index==2:
    print("Stomach")
    print(food[2])
        
        
if results_index==3:
    print("Heart")
    print(food[3])
    
    
    
    
  
