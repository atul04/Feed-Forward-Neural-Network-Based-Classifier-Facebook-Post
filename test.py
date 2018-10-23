# @Author: Atul Sahay <atul>
# @Date:   2018-10-22T18:34:28+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2018-10-23T18:47:09+05:30

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import math


#Xor data
XORdata=np.array([[0,0,0],[0,1,1],[1,0,2],[1,1,3]])
X=XORdata[:,0:2]
y=XORdata[:,-1]

def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)



def initialize_network():

    input_neurons=len(X[0])
    hidden_neurons=input_neurons+1
    output_neurons=8

    n_hidden_layers=1

    net=list()

    for h in range(n_hidden_layers):
        if h!=0:
            input_neurons=len(net[-1])

        hidden_layer = [ { 'weights': np.random.uniform(size=input_neurons)} for i in range(hidden_neurons) ]
        net.append(hidden_layer)

    output_layer = [ { 'weights': np.random.uniform(size=hidden_neurons)} for i in range(output_neurons)]
    net.append(output_layer)

    return net

################### Initialization of network
net = initialize_network()
print_network(net)


def activate_sigmoid(sum):
    return (1/(1+np.exp(-sum)))

def forward_propagation(net,input):
    row=input
    for layer in net:
        prev_input=np.array([])
        for neuron in layer:
            sum=neuron['weights'].T.dot(row)

            result=activate_sigmoid(sum)
            neuron['result']=result

            prev_input=np.append(prev_input,[result])
        row =prev_input

    return row

def sigmoidDerivative(output):
    return output*(1.0-output)



################# Need to perform square loss

def back_propagation(net,row,expected):
     for i in reversed(range(len(net))):
            layer=net[i]
            errors=np.array([])
            if i==len(net)-1:
                results=[neuron['result'] for neuron in layer]
                errors = expected-np.array(results)
            else:
                for j in range(len(layer)):
                    herror=0
                    nextlayer=net[i+1]
                    for neuron in nextlayer:
                        herror+=(neuron['weights'][j]*neuron['delta'])
                    errors=np.append(errors,[herror])

            for j in range(len(layer)):
                neuron=layer[j]
                neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])


def updateWeights(net,input,lrate):

    for i in range(len(net)):
        inputs = input
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]

def training(net, epochs,lrate,n_outputs):
    errors=[]
    for epoch in range(epochs):
        sum_error=0
        for i,row in enumerate(X):
            outputs=forward_propagation(net,row)

            # expected=[0.0 for i in range(n_outputs)]
            # expected[y[i]]=1

            expected = np.unpackbits(np.uint8(y[i]))

            sum_error+=sum([(expected[j]-outputs[j])**2 for j in range(len(expected))])
            back_propagation(net,row,expected)
            updateWeights(net,row,0.05)
        if epoch%10000 ==0:
            print('>epoch=%d,error=%.3f'%(epoch,sum_error))
            errors.append(sum_error)
    return errors

errors=training(net,100000, 0.05,2)

epochs=[0,1,2,3,4,5,6,7,8,9]
plt.plot(epochs,errors)
plt.xlabel("epochs in 10000's")
plt.ylabel('error')
plt.show()

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagation(net, row)
    return outputs



pred=predict(net,np.array([0,1]))
#output=np.argmax(pred)
super_threshold_indices = pred >= 0.5
pred[super_threshold_indices] = 1
super_threshold_indices2 = pred < 0.5
pred[super_threshold_indices2] = 0
pred = pred.astype('int')
output = np.packbits(pred)
print(output)


print_network(net)
