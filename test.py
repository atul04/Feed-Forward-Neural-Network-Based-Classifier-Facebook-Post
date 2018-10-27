# @Author: Atul Sahay <atul>
# @Date:   2018-10-22T18:34:28+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2018-10-27T16:44:19+05:30

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import math


HIDDEN_LAYERS = 2
HIDDEN_UNITS = 100
OUTPUT_UNITS = 3
BATCH_SIZE = 200
################ For One Hot encoding of the values ##########################

def one_hot_encode(num,size=OUTPUT_UNITS):
    arr = np.zeros(size)
    np.put(arr, num-1, 1)
    return arr

def one_hot_decode(arr):
    return np.where(arr==1)[0][0]+1
############### End One Hot encoding ########################################


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

def print_file(f,net):
    for i,layer in enumerate(net,1):
        f.write("Layer {} \n".format(i))
        for j,neuron in enumerate(layer,1):
            f.write("neuron {} : {}\n".format(j,neuron))

def initialize_network(X,n_o_neurons,hidden_units,n_h_layers):
    print(X.ix[0])
    input_neurons=len(X.ix[0])
    hidden_neurons=hidden_units
    output_neurons=n_o_neurons

    n_hidden_layers=n_h_layers

    net=list()

    for h in range(n_hidden_layers):
        if h!=0:
            input_neurons=len(net[-1])
                                                        #'''size=input_neurons'''
        hidden_layer = [ { 'weights': np.random.uniform(-1,1,size=input_neurons)} for i in range(hidden_neurons) ]
        net.append(hidden_layer)
                                                        #'''size=hidden_neurons'''
    output_layer = [ { 'weights': np.random.uniform(-1,1,size=hidden_neurons)} for i in range(output_neurons)]
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
        # print(layer)
        neuron = []
        for n in layer:
            neuron.append(n['weights'])
        # print("here is the neuron")
        neuron = np.array(neuron)
        # print(neuron.shape)

        # print("\nAnd here is the transpose")
        # neuronT = neuron.T
        # print(neuronT.shape)

        sum = neuron.dot(row)
        # print(sum)
        result = activate_sigmoid(sum)
        # print(result)

        for index,neuron in enumerate(layer):
            neuron['result'] = result[index]
        # print(layer)

        row = result
        # for neuron in layer:
        #     # print(neuron['weights'])
        #     # print(row)
        #     sum=neuron['weights'].T.dot(row)
        #
        #     result=activate_sigmoid(sum)
        #     neuron['result']=result
        #
        #     prev_input=np.append(prev_input,[result])
        # row =prev_input

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
                # print("errors[j]",errors[j])
                # print("result",neuron['result'])
                neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])
                # print(neuron['delta'])

def updateWeights(net,input,lrate):
    # print_network(net)
    for i in range(len(net)):
        inputs = input
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]

def get_mini_batch(batch_size,X,y):
    indices = list(np.random.randint(0,len(X),batch_size))
    x_mini = []
    y_mini = np.array([])
    for i in indices:
        x_mini.append(X[i])
        y_mini = np.append(y_mini,y[i])
    x_mini = np.array(x_mini)
    print(x_mini)
    print(y_mini)

    return x_mini,y_mini

def training(X,net, epochs,lrate,y,batch_size):
    errors=[]
    for epoch in range(epochs):
        sum_error=0
        x_mini,y_mini = get_mini_batch(batch_size,X,y)
        for i,row in enumerate(x_mini):
            # print(i)
            outputs=forward_propagation(net,row)
            # print(outputs)
            # expected=[0.0 for i in range(n_outputs)]
            # expected[y[i]]=1

            # expected = np.unpackbits(np.uint8(y_mini[i]))
            expected = one_hot_encode(y_mini[i])

            # sum_error_back += (expected-outputs)
            sum_error += np.sum((expected-outputs)**2)
            # print(sum_error)
            # if(i%100==0):
            back_propagation(net,row,expected)
            updateWeights(net,row,lrate)

        # print("sum_error_back",sum_error_back)
        # sum_error_back/=100
        # if epoch%10 ==0:
            if(i%99==0):
                print("expected=",expected)
                print("output=",outputs)
                print("Sum_error=",sum_error)


        print('>epoch=%d,error=%f'%(epoch,sum_error))
        errors.append(sum_error)
        # sum_error_back = 0
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
    outputs = forward_propagation(network, row)
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
    indexes = int(0.70*x_train.shape[0])
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

    net = initialize_network(x_train,OUTPUT_UNITS,HIDDEN_UNITS,HIDDEN_LAYERS)
    print_network(net)
    ############ Training of the network ###################3
    errors=training(x_train.values,net,500, 0.01,y_train.values,BATCH_SIZE)
    print_network(net)
    epochs=[ i for i in range(len(errors)) ]
    plt.plot(epochs,errors)
    plt.xlabel("epochs in 1's")
    plt.ylabel('error')
    plt.show()

    ########### Prediction ###################################
    f = open('result.txt','a')
    f.write('\n\nResult: {} {} Training Loss: {}\n\n'.format(HIDDEN_LAYERS,HIDDEN_UNITS,errors[-1]))
    square_loss =0
    for i in range(len(x_valid.values)):
            # print(x_valid.values[i])
            # print(y_valid.values[i])
            pred=predict(net,x_valid.values[i])
            # print(pred)
            #output=np.argmax(pred)
            # super_threshold_indices = pred >= 0.5
            # pred[super_threshold_indices] = 1
            # super_threshold_indices2 = pred < 0.5
            # pred[super_threshold_indices2] = 0
            pred = one_hot_encode(np.argmax(pred)+1)
            # output = np.packbits(pred)[0]
            print(pred)
            output = one_hot_decode(pred)
            square_loss+=(output-y_valid.values[i])**2
            print("expected : {} actual : {}".format(y_valid.values[i],output))
            f.write("expected : {} actual : {}\n".format(y_valid.values[i],output))
    print(square_loss,len(x_valid.values))
    square_loss = square_loss*1.0 / float(len(x_valid.values))
    print("Square_loss {} ".format(square_loss))
    f.write("\n\nSquare_loss {} \n\n".format(square_loss))
    f.close()

    f = open('net.txt','a')
    f.write('\n\nNew Net\n\n')
    print_file(f,net)
    f.close()

    # print_network(net)





#################### Driver Function
if __name__ == '__main__':
    main()
