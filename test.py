# @Author: Atul Sahay <atul>
# @Date:   2018-10-22T18:34:28+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2018-10-29T18:21:47+05:30

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import math


HIDDEN_LAYERS = 1
HIDDEN_UNITS = 100
OUTPUT_UNITS = 3
BATCH_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 1000
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
    data = data.sample(frac=1).reset_index(drop=True)
    x_train = data.iloc[:,:-1]
    y_train = data.iloc[:,-1]

    # print(x_train)
    # print(y_train)
    # exit()
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
        # neuron = np.squeeze(neuron)
        # print(neuron.shape)
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
                # for mse###errors = expected-np.array(results)
                errors = np.array(results) - expected
                for j in range(len(layer)):
                    neuron=layer[j]
                    # print("errors[j]",errors[j])
                    # print("result",neuron['result'])
                    #for mse### neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])
                    neuron['delta']=errors[j]
                    # print(neuron['delta'])
            else:
                # for j in range(len(layer)):
                #     herror=0
                #     nextlayer=net[i+1]
                #     for neuron in nextlayer:
                #         herror+=(neuron['weights'][j]*neuron['delta'])
                #     errors=np.append(errors,[herror])
                nextlayer = net[i+1]
                delta = np.array([])
                errors = []
                for neuron in nextlayer:
                    # print(neuron['weights'])
                    errors.append(neuron['weights'])
                    delta = np.append(delta,neuron['delta'])
                errors = np.array(errors)
                # errors = np.squeeze(errors)
                # print(errors.shape,delta.shape)
                errors = np.dot(errors.T,delta)
                # print(errors.shape)
                # exit()
                results=[neuron['result'] for neuron in layer]
                results = np.array(results)
                # print(errors.shape,results.shape)
                final_error = errors*results
                # print(final_error.shape)
                # exit()
                for j in range(len(layer)):
                    neuron=layer[j]
                    # print("errors[j]",errors[j])
                    # print("result",neuron['result'])
                    #for mse### neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])
                    neuron['delta']=final_error[j]
                    # print(neuron['delta'])


def updateWeights(net,input,lrate):
    # print_network(net)
    for i in range(len(net)):
        inputs = input
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]
        inputs = np.asmatrix(inputs).T
        delta = [neuron['delta'] for neuron in net[i]]
        delta = np.asmatrix(delta).T
        # print(delta.shape,inputs.shape,inputs.T.shape)
        error_loss = lrate*np.dot(delta,inputs.T)
        # print(error_loss.shape)
        weights = [neuron['weights'] for neuron in net[i]]
        weights = np.asmatrix(weights)
        # print(weights.shape)
        weights = weights - error_loss
        # print(weights.shape)
        # exit()
        for index,neuron in enumerate(net[i]):
            # print(np)
            # print(weights[index].shape)
            c = np.ravel(weights[index])
            neuron['weights']=c

def get_mini_batch(batch_size,X,y):
    indices = list(np.random.randint(0,len(X),batch_size))
    # x_mini = []
    # y_mini = np.array([])
    # for i in indices:
    #     x_mini.append(X[i])
    #     y_mini = np.append(y_mini,y[i])
    # x_mini = np.array(x_mini)
    # print(x_mini)
    # print(y_mini)
    x_mini = np.array([])
    y_mini = np.array([])
    x_mini = X[indices]
    y_mini = y[indices]
    # print(x_mini)
    # print(y_mini)
    # exit()
    return x_mini,y_mini

def cross_entropy(expected,output):
    result = -(expected*np.log(output)+(1-expected)*np.log(1-output))
    return result

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
            sum_error += np.sum(cross_entropy(expected,outputs))/OUTPUT_UNITS
            # print(sum_error)
            # if(i%100==0):
            back_propagation(net,row,expected)
            updateWeights(net,row,lrate)

        # print("sum_error_back",sum_error_back)
        # sum_error_back/=100
        # if epoch%10 ==0:
            if(i%(BATCH_SIZE-1)==0):
                print("expected=",expected)
                print("output=",outputs)
                print("Sum_error=",sum_error/BATCH_SIZE)

        sum_error/=BATCH_SIZE
        print('>epoch=%d,error=%f'%(epoch,sum_error))
        errors.append(sum_error)
    mean_cross = np.mean(errors)
        # sum_error_back = 0
    return errors,mean_cross

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

def generate_output(x_test, net):
# 	# writes a file (output.csv) containing target variables in required format for Kaggle Submission.
    print("Generating the output file:--")
    df = pd.DataFrame(columns=['predicted_class'])
    # y_P_list = []
    # idList = [ i for i in range(int(len(phi_test)))]
    for i in range(int(len(x_test))):
        # print(phi_test[i])
        y_pred=predict(net,x_test[i])
        # print(pred)
        #output=np.argmax(pred)
        # super_threshold_indices = pred >= 0.5
        # pred[super_threshold_indices] = 1
        # super_threshold_indices2 = pred < 0.5
        # pred[super_threshold_indices2] = 0
        # print(target)
        # print(pred)
        y_pred = np.argmax(y_pred)+1
        df.loc[i+1] = np.array([y_pred])
    txt = 'output_H_'+str(HIDDEN_LAYERS)+'_L_'+str(LEARNING_RATE)+'_U_'+str(HIDDEN_UNITS)+'.txt'
    df.to_csv(txt)
    print("Done")


def total_loss(net,x,y):
    sum_error=0
    for i,row in enumerate(x):
        outputs=forward_propagation(net,row)
        expected = one_hot_encode(y[i])
        sum_error += np.sum(cross_entropy(expected,outputs))/OUTPUT_UNITS
    sum_error/=len(x)
    return sum_error

def total_accuracy(net,x,y):
    count = 0
    for i,row in enumerate(x):
        outputs=forward_propagation(net,row)
        pred = np.argmax(outputs)+1
        if(pred==y[i]):
            count +=1
    count/=len(x)
    return count





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
    # print_network(net)
    ############ Training of the network ###################3
    errors,mean_cross=training(x_train.values,net,EPOCHS, LEARNING_RATE,y_train.values,BATCH_SIZE)
    print_network(net)
    epochs=[ i for i in range(len(errors)) ]
    plt.plot(epochs,errors)
    plt.xlabel("Epochs [Batches("+str(BATCH_SIZE)+"'s)] ")
    plt.ylabel('error')
    plt.show()

    print("Calculating: Loss and Accuracy")
    train_loss,train_acc = total_loss(net,x_train.values,y_train.values),total_accuracy(net,x_train.values,y_train.values)
    val_loss,val_acc = total_loss(net,x_valid.values,y_valid.values),total_accuracy(net,x_valid.values,y_valid.values)
    print("Train : Loss {} Acc {} ".format(train_loss,train_acc))
    print("Valid : Loss {} Acc {} ".format(val_loss,val_acc))

    ########### Prediction ###################################
    txt = 'result_cross_H_'+str(HIDDEN_LAYERS)+'_L_'+str(LEARNING_RATE)+'_U_'+str(HIDDEN_UNITS)+'.txt'
    f = open(txt,'a')
    f.write('\n\nResult: Hidden layers {} H_units {} Lrate {} Epochs {} Batch Size {} Training Loss: {}\n\n'.format(HIDDEN_LAYERS,HIDDEN_UNITS,LEARNING_RATE,EPOCHS,BATCH_SIZE,mean_cross))
    f.write('\nTrain : Loss {} Acc {} \n'.format(train_loss,train_acc))
    f.write('\nValid : Loss {} Acc {} \n\n'.format(val_loss,val_acc))
    cross_entropy_loss =0
    for i in range(len(x_valid.values)):
            # print(x_valid.values[i])
            # print(y_valid.values[i])
            pred=predict(net,x_valid.values[i])

            #output=np.argmax(pred)
            # super_threshold_indices = pred >= 0.5
            # pred[super_threshold_indices] = 1
            # super_threshold_indices2 = pred < 0.5
            # pred[super_threshold_indices2] = 0
            # print(pred)
            target=one_hot_encode(y_valid.values[i])
            # print(target)
            # print(pred)
            cross_entropy_loss+=np.mean(cross_entropy(target,pred))
            # print(cross_entropy_loss)
            pred = one_hot_encode(np.argmax(pred)+1)
            # output = np.packbits(pred)[0]
            #output = one_hot_decode(pred)
            output=pred

            #square_loss+=(output-y_valid.values[i])**2
            print(output)
            print(target)

            # cross_entropy_loss+=np.mean(cross_entropy(target,output))
            print("expected : {} actual : {}".format(y_valid.values[i],one_hot_decode(output)))
            f.write("expected : {} actual : {}\n".format(y_valid.values[i],one_hot_decode(output)))
    print(cross_entropy_loss,len(x_valid.values))
    cross_entropy_loss = cross_entropy_loss*1.0 / float(len(x_valid.values))
    print("cross_entropy_loss {} ".format(cross_entropy_loss))
    f.write("\n\ncross_entropy_loss {} \n\n".format(cross_entropy_loss))
    f.close()
    f = open('net_cross.txt','a')
    f.write('\n\nNew Net\n\n')
    print_file(f,net)
    f.close()

    # Output Generation is done
    generate_output(x_test.values,net)
    # print_network(net)





#################### Driver Function
if __name__ == '__main__':
    main()
