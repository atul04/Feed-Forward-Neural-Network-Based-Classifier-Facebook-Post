# @Author: Atul Sahay <atul>
# @Date:   2018-10-22T18:34:28+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2018-11-04T01:30:56+05:30

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import math

HIDDEN_LAYERS = 2
HIDDEN_UNITS = 100
OUTPUT_UNITS = 3
BATCH_SIZE = 100
LAMBDA = 4
LEARNING_RATE = 0.1
EPOCHS = 1000
ACTIVATION_FUNCTIONS = ["sigmoid","tanh","relu","softplus"]
ACTIVATION = ACTIVATION_FUNCTIONS[0]
EPSILON = 0.000000000001
DROPOUT = 0
DECAY = False
PARTITION = 0.90
n_weights_count = 0

###########################################################################################################
#################### SETS METHODS FOR EXTRACTING FEATURES SET FROM THE DATA SET ##########################
###########################################################################################################
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


###########################################################################################################
#################### END OF FEATURE EXTRACTION  ##########################
#########################################################################################################


################ For One Hot encoding of the values (OUTPUT) ##########################

def one_hot_encode(num,size=OUTPUT_UNITS):
    arr = np.zeros(size)
    np.put(arr, num-1, 1)
    return arr

def one_hot_decode(arr):
    return np.where(arr==1)[0][0]+1
############### End One Hot encoding ########################################


########## HELPING METHOD FOR PRINTING NETWORK CONFIGURATION AT ############
#########  COMMAND PROMPT OR SOME FILE                         ############
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

################################ DONE ####################################


# METHOD TO INITIALIZE THE NETWORK############################
def initialize_network(X,n_o_neurons,hidden_units,n_h_layers):
    global n_weights_count
    # print(X.ix[0])
    input_neurons=len(X.ix[0])
    hidden_neurons=hidden_units
    output_neurons=n_o_neurons

    n_hidden_layers=n_h_layers

    net=list()
    n_weights_count=0

    for h in range(n_hidden_layers):
        if h!=0:
            input_neurons=len(net[-1])
        n_weights_count+=(input_neurons)*hidden_neurons
                                                    #'''size=input_neurons'''
        hidden_layer = [ { 'weights': np.random.uniform(-1,1,size=input_neurons)} for i in range(hidden_neurons) ]
        net.append(hidden_layer)
    n_weights_count+=(hidden_neurons)*output_neurons
                                                        #'''size=hidden_neurons'''
    output_layer = [ { 'weights': np.random.uniform(-1,1,size=hidden_neurons)} for i in range(output_neurons)]
    net.append(output_layer)

    return net


############################### ACTIVATION FUNCTIONS IMPLEMENTATION ####################################

def activate_sigmoid(sum):
    return (1/(1+np.exp(-sum)))

def activate_tanh(sum):
    return np.tanh(sum)

def activate_relu(sum):
    return sum.clip(0)

def activate_softplus(sum):
    return np.log(1+np.exp(sum))


############################# DONE WITH ACTIVATION METHODS ##############################################

############################# DERIVATIVES OF EACH ACTIVATION FUNCTION ###################################
def sigmoidDerivative(output):
    return output*(1.0-output)

def tanhDerivative(output):
    return (1-output**2)

def reluDerivative(output):
    der = output.copy()
    der[der>0] = 1
    der[der<0] = 0
    return der

def softplusDerivative(output):
    return activate_sigmoid(output)
########################### DONE WITH DERIVATIVES #######################################################

############################ DROPUT METHOD IMPLEMENTATION TAKING DROPUT PARAM AS GLOBAL ################
def dropout_activation(activation):
    # print(activation)
    global DROPOUT
    size = len(activation)
    # print(size)
    indices = np.random.choice(np.arange(0,size),int(size*DROPOUT),replace=False)
    # print(indices)
    activation[indices] = EPSILON
    # print(activation)
    return activation

######################### RETURNING NEW ACTIVATION LIST ################################################


######################## FORWARD PROPAGATION IMPLEMENTATION ###########################################
def forward_propagation(net,input):
    global ACTIVATION
    row=input
    for index,layer in enumerate(net):
        prev_input=np.array([])
        # print(layer)
        # print("layer ",index+1)
        neuron = []
        for n in layer:
            neuron.append(n['weights'])
        # print("here is the neuron")
        neuron = np.array(neuron)

        sum = neuron.dot(row)
        # print(sum)
        if(ACTIVATION=="sigmoid" or index == (len(net)-1)):
            result = activate_sigmoid(sum)
        elif(ACTIVATION=="tanh"):
            result = activate_tanh(sum)
        elif(ACTIVATION=="relu"):
            result = activate_relu(sum)
        elif(ACTIVATION=="softplus"):
            result = activate_softplus(sum)
        # print(result)
        result = dropout_activation(result)

        for index,neuron in enumerate(layer):
            neuron['result'] = result[index]
        # print(layer)

        row = result
    return row
###########################################################################


################# BACK PROPAGATION IMPLEMENTATION ##############################

def back_propagation(net,row,expected):
    global ACTIVATION
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
            if(ACTIVATION == "sigmoid"):
                final_error = errors*sigmoidDerivative(results)
            elif(ACTIVATION == "tanh"):
                final_error = errors*tanhDerivative(results)
            elif(ACTIVATION == "relu"):
                final_error = errors*reluDerivative(results)
            elif(ACTIVATION == "softplus"):
                final_error = errors*softplusDerivative(results)
            # print(final_error.shape)
            # exit()
            for j in range(len(layer)):
                neuron=layer[j]
                neuron['delta']=final_error[j]
                # print(neuron['delta'])


############################## WEIGHT UPDATATION ########################
def updateWeights(net,input,lrate,lam):
    # print_network(net)
    global n_weights_count
    # print(n_weights_count)
    reg = (1-lrate*lam/n_weights_count)
    # print("reg ",reg)
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
        weights = np.array(weights)
        # print("weights :",weights)
        weights = weights*reg
        # print("weights :",weights)
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


##################### GET MINI BATCH STOCHASTIC APPROACH ###########################
def get_mini_batch(batch_size,X,y):
    indices = list(np.random.randint(0,len(X),batch_size))
    x_mini = np.array([])
    y_mini = np.array([])
    x_mini = X[indices]
    y_mini = y[indices]
    return x_mini,y_mini

################### CROSS ENTROPY LOSS IMPLEMENTATION WITH EPSILON FOR LOG(0) CASE ###########
def cross_entropy(expected,output):
    global EPSILON
    result = -(expected*np.log(output+EPSILON)+(1-expected)*np.log(1-output+EPSILON))
    return result


################ L-2 NORM REGULARIZATION IMPLEMENTATION ##############################
def regularization(net,lam):
    neuron_weights_mean = []
    for i in range(len(net)):
        layer = net[i]
        for neuron in layer:
            c = neuron['weights']
            c = np.mean(c**2)
            neuron_weights_mean.append(c)
    neuron_weights_mean = np.array(neuron_weights_mean)
    neuron_weights_mean = np.mean(neuron_weights_mean)

    reg = lam*neuron_weights_mean

    return reg


############ DECAY LEARNING RATE IMPLEMMENTATION #########################################
def decay_learn(loss,lrate,min_loss):
    if min_loss is None:
        return lrate, loss
    if(loss<min_loss):
        lrate=lrate/(1+np.abs(min_loss-loss))
        min_loss = loss
    return lrate,min_loss


############## TRAINGING METHOD IMPLEMENTATION #########################################
def training(X,net, epochs,lrate,y,batch_size,lam,x_valid,y_valid):
    errors=[]

    min_loss = None
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
            updateWeights(net,row,lrate,lam)
        # print("sum_error_back",sum_error_back)
        # sum_error_back/=100
        # if epoch%10 ==0:
            if(i%(BATCH_SIZE-1)==0):
                print("expected=",expected)
                print("output=",outputs)
                print("Sum_error=",sum_error/BATCH_SIZE)

        if(DECAY):
            loss = total_loss(net,x_valid,y_valid,lam)
            lrate,min_loss = decay_learn(loss,lrate,min_loss)
        # print("loss : {} LRate : {}".format(loss,lrate))
        sum_error/=BATCH_SIZE
        print('>epoch=%d,error=%f'%(epoch,sum_error))
        errors.append(sum_error)
    mean_cross = np.mean(errors)
        # sum_error_back = 0
    return errors, mean_cross

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagation(network, row)
    return outputs



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
    txt = 'output_H_'+str(HIDDEN_LAYERS)+'_L_'+str(LEARNING_RATE)+'_U_'+str(HIDDEN_UNITS)+'_lam_'+str(LAMBDA)+'_ACTIVATION_'+str(ACTIVATION)+'_Drop_'+str(DROPOUT)+'.txt'
    df.to_csv(txt)
    print("Done")


def total_loss(net,x,y,lam):
    global n_weights_count
    reg = regularization(net,lam)
    sum_error=0
    for i,row in enumerate(x):
        outputs=forward_propagation(net,row)
        expected = one_hot_encode(y[i])
        sum_error += np.sum(cross_entropy(expected,outputs))/OUTPUT_UNITS
    sum_error/=len(x)
    reg/=n_weights_count
    # print("reg",reg)
    t_error = sum_error + reg
    return t_error

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
    # global HIDDEN_LAYERS, HIDDEN_UNITS, OUTPUT_UNITS, BATCH_SIZE, LAMBDA, LEARNING_RATE, ACTIVATION, ACTIVATION_FUNCTIONS, DROPOUT, DECAY, PARTITION

    # print(HIDDEN_LAYERS, HIDDEN_UNITS, OUTPUT_UNITS, BATCH_SIZE, LAMBDA, LEARNING_RATE, ACTIVATION, ACTIVATION_FUNCTIONS, DROPOUT, DECAY, PARTITION)
    # exit()
    """
    Calls functions required to do tasks in sequence
    say :
    	train_file = first_argument
    	test_file = second_argument
    	x_train, y_train = get_features();
    	task1();task2();task3();.....
    """
    global train_mean, train_std, n_weights_count
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
    indexes = int(PARTITION*x_train.shape[0])
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
    print("#Weights :",n_weights_count)
    ############ Training of the network ###################3
    errors,mean_cross=training(x_train.values,net,EPOCHS, LEARNING_RATE,y_train.values,BATCH_SIZE,LAMBDA,x_valid.values,y_valid.values)
    # print_network(net)
    # epochs=[ i for i in range(len(errors)) ]
    # plt.plot(epochs,errors)
    # plt.xlabel("Epochs [Batches("+str(BATCH_SIZE)+"'s)] ")
    # plt.ylabel('error')
    # plt.show()

    print("Calculating: Loss and Accuracy")
    train_loss,train_acc = total_loss(net,x_train.values,y_train.values,LAMBDA),total_accuracy(net,x_train.values,y_train.values)
    val_loss,val_acc = total_loss(net,x_valid.values,y_valid.values,LAMBDA),total_accuracy(net,x_valid.values,y_valid.values)
    print("Train : Loss {} Acc {} ".format(train_loss,train_acc))
    print("Valid : Loss {} Acc {} ".format(val_loss,val_acc))

    # ########### Prediction ###################################
    # txt = 'result_cross_H_'+str(HIDDEN_LAYERS)+'_L_'+str(LEARNING_RATE)+'_U_'+str(HIDDEN_UNITS)+'_lam_'+str(LAMBDA)+'_ACTIVATION_'+str(ACTIVATION)+'_Drop_'+str(DROPOUT)+'.txt'
    # f = open(txt,'a')
    # f.write('\n\nResult: Activation {} Hidden layers {} H_units {} Lrate {} LAMBDA {} Epochs {} Batch Size {} Drop out {} Training Loss: {}\n\n'.format(ACTIVATION,HIDDEN_LAYERS,HIDDEN_UNITS,LEARNING_RATE,LAMBDA,EPOCHS,BATCH_SIZE,DROPOUT,mean_cross))
    # f.write('\nTrain : Loss {} Acc {} \n'.format(train_loss,train_acc))
    # f.write('\nValid : Loss {} Acc {} \n\n'.format(val_loss,val_acc))
    # cross_entropy_loss =0
    # for i in range(len(x_valid.values)):
    #         # print(x_valid.values[i])
    #         # print(y_valid.values[i])
    #         pred=predict(net,x_valid.values[i])
    #
    #         #output=np.argmax(pred)
    #         # super_threshold_indices = pred >= 0.5
    #         # pred[super_threshold_indices] = 1
    #         # super_threshold_indices2 = pred < 0.5
    #         # pred[super_threshold_indices2] = 0
    #         # print(pred)
    #         target=one_hot_encode(y_valid.values[i])
    #         # print(target)
    #         # print(pred)
    #         cross_entropy_loss+=np.mean(cross_entropy(target,pred))
    #         # print(cross_entropy_loss)
    #         pred = one_hot_encode(np.argmax(pred)+1)
    #         # output = np.packbits(pred)[0]
    #         #output = one_hot_decode(pred)
    #         output=pred
    #
    #         #square_loss+=(output-y_valid.values[i])**2
    #         print(output)
    #         print(target)
    #
    #         # cross_entropy_loss+=np.mean(cross_entropy(target,output))
    #         print("expected : {} actual : {}".format(y_valid.values[i],one_hot_decode(output)))
    #         f.write("expected : {} actual : {}\n".format(y_valid.values[i],one_hot_decode(output)))
    # # print(cross_entropy_loss,len(x_valid.values))
    # cross_entropy_loss = cross_entropy_loss*1.0 / float(len(x_valid.values))
    # # print("cross_entropy_loss {} ".format(cross_entropy_loss))
    # # f.write("\n\ncross_entropy_loss {} \n\n".format(cross_entropy_loss))
    # f.close()
    # f = open('net_cross.txt','a')
    # f.write('\n\nNew Net\n\n')
    # print_file(f,net)
    # f.close()
    #
    # # Output Generation is done
    generate_output(x_test.values,net)
    # print_network(net)





#################### Driver Function
if __name__ == '__main__':
    # global HIDDEN_LAYERS, HIDDEN_UNITS, OUTPUT_UNITS, BATCH_SIZE, LAMBDA, LEARNING_RATE, ACTIVATION, ACTIVATION_FUNCTIONS, DROPOUT, DECAY, PARTITION
    print("******************Neural Network Model***********************")
    print("Enter model configuration")
    HIDDEN_LAYERS = int(input("Hidden Layers : "))
    HIDDEN_UNITS  = int(input("Hidden Units : "))
    print("Activation Functions:  \n")
    print("1.Sigmoid\n")
    print("2.Tanh\n")
    print("3.ReLu\n")
    print("4.Softplus\n")
    choice = int(input("Choice [1,2,3,4] : ")) - 1
    ACTIVATION = ACTIVATION_FUNCTIONS[choice]
    LEARNING_RATE = float(input("Enter learning rate : "))
    print("Want to decay on learning rate : \n")
    print("1. True\n")
    print("2. False\n")
    choice = int(input("Choice (1 or 2): "))
    DECAY = True if choice == 1 else False
    c = float(input("VALIDATION SET PARTITION (%): "))
    PARTITION = c/100
    BATCH_SIZE = int(input("Enter Batch size: "))
    print("For regualrization \n")
    LAMBDA = int(input("Enter Lambda (if not enter 0): "))
    c =  float(input("Enter DROPOUT (%) : "))
    DROPUT = c/100
    main()
