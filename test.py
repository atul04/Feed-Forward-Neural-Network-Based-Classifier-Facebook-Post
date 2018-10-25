# @Author: Atul Sahay <atul>
# @Date:   2018-10-22T18:34:28+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2018-10-25T15:42:41+05:30

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import math



def to_map(data_set):
    data_set = pd.concat([data_set,pd.get_dummies(data_set['post_day'], prefix='post_day')],axis=1)

    # now drop the original
    data_set.drop(['post_day'],axis=1, inplace=True)

    data_set = pd.concat([data_set,pd.get_dummies(data_set['basetime_day'], prefix='basetime_day')],axis=1)

    # now drop the original
    data_set.drop(['basetime_day'],axis=1, inplace=True)

    return data_set

# Feature scaling , here I have used min_max
def to_normalize(data_set):
    global train_mean, train_std
    train_mean = data_set.mean()
    train_std = data_set.std()
    data_set = (data_set - data_set.min())/(data_set.max() - data_set.min())
    return data_set

# Split in x and y
def split(data):
    x_train = data.iloc[:,:-1]
    y_train = data.iloc[:,-1]

    return x_train, y_train

# Provide features set and target set
def get_features(file_path):
	# Given a file path , return feature matrix and target labels
    data = pd.read_csv(file_path)
    return split(data)








#Xor data
XORdata=np.array([[0,0,0],[0,1,1],[1,0,2],[1,1,3]])
X=XORdata[:,0:2]
y=XORdata[:,-1]

def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)



def initialize_network(X,n_o_neurons,hidden_units):
    print(X.ix[0])
    input_neurons=len(X.ix[0])
    hidden_neurons=hidden_units
    output_neurons=n_o_neurons

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
# net = initialize_network(X,8,len(X[0])+1)
# print_network(net)


def activate_sigmoid(sum):
    return (1/(1+np.exp(-sum)))

def forward_propagation(net,input):
    row=input
    for layer in net:
        prev_input=np.array([])
        for neuron in layer:
            # print(neuron['weights'])
            # print(row)
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

def training(X,net, epochs,lrate,y):
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

# errors=training(net,100000, 0.05,2,y)
#
# epochs=[0,1,2,3,4,5,6,7,8,9]
# plt.plot(epochs,errors)
# plt.xlabel("epochs in 10000's")
# plt.ylabel('error')
# plt.show()

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagation(net, row)
    return outputs



# pred=predict(net,np.array([0,1]))
# #output=np.argmax(pred)
# super_threshold_indices = pred >= 0.5
# pred[super_threshold_indices] = 1
# super_threshold_indices2 = pred < 0.5
# pred[super_threshold_indices2] = 0
# pred = pred.astype('int')
# output = np.packbits(pred)
# print(output)
#
#
# print_network(net)







def main():
    """
    Calls functions required to do tasks in sequence
    say :
    	train_file = first_argument
    	test_file = second_argument
    	x_train, y_train = get_features();
    	task1();task2();task3();.....
    """
    global train_mean, train_std
    # pow = 1
    train_file = sys.argv[1]
    # train_file = '/home/atul/college/cs725/Assignment/train.csv'
    test_file = sys.argv[2]
    # test_file = '/home/atul/college/cs725/Assignment/test.csv'
    print("Reading Files...")
    x_test = pd.read_csv(test_file)

    x_train, y_train = get_features(train_file)
    print("Done")
    ################################## Mapping days to one hot vector################
    x_train = to_map(x_train)
    x_test = to_map(x_test)
    ##############################################################################

    ####################### Normalizing the data points##############################
    x_train = to_normalize(x_train)

    # To take validation set out in proportion of 20-80 #############################
    indexes = int(0.80*x_train.shape[0])
    x_train, x_valid = x_train.iloc[:indexes], x_train.iloc[indexes:]
    y_train, y_valid = y_train.iloc[:indexes], y_train.iloc[indexes:]

    ####################### Normalizing the data points##############################
    x_test = to_normalize(x_test)
    x_test.promotion = x_test.promotion.fillna(0)


    ####################################################

    #Appending a series of Ones for bias in x_train
    ones = np.ones(x_train.shape[0])
    x_train.insert(loc=x_train.shape[1], column='Ones', value=ones)

    #Appending a series of Ones for bias in x_valid
    ones = np.ones(x_valid.shape[0])
    x_valid.insert(loc=x_valid.shape[1], column='Ones', value=ones)

    #Appending a series of ones for bias in x_test
    ones = np.ones(x_test.shape[0])
    x_test.insert(loc=x_test.shape[1], column='Ones', value=ones)
################################################################

    #### Initialization of network ############################

    net = initialize_network(x_train,8,100)
    print_network(net)

    ############ Training of the network ###################3
    errors=training(x_train.values,net,100000, 0.05,y_train.values)

    epochs=[0,1,2,3,4,5,6,7,8,9]
    plt.plot(epochs,errors)
    plt.xlabel("epochs in 10000's")
    plt.ylabel('error')
    plt.show()

    ########### Prediction ###################################

    pred=predict(net,x_valid.ix[3])
    #output=np.argmax(pred)
    super_threshold_indices = pred >= 0.5
    pred[super_threshold_indices] = 1
    super_threshold_indices2 = pred < 0.5
    pred[super_threshold_indices2] = 0
    pred = pred.astype('int')
    output = np.packbits(pred)
    print("expected : {} actual : {}".format(y_valid.ix[3],output))

    print_network(net)





#################### Driver Function
if __name__ == '__main__':
    main()
