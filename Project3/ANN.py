import numpy as np
from matplotlib import pyplot as plt
import random
"""
This class calculates the ANN for given input and output
Author: Turgut Alp Edis
"""
class ANN:

    #ANN with one hidden layer
    #3 weight arrays for input, hidden layer and output
    def __init__(self, node_cnt):
        #Initially arrange the weights randomly, then rearrange them according to the error
        self.__input_weight = np.array( [ random.random() for i in range(node_cnt) ] )
        self.__hidden_weight = np.array( [ random.random() for i in range(node_cnt) ] )
        self.__output_weight = 1 / float( node_cnt )

    #Activation function
    def sigmoid(self, value):
        sigmoid = 1 / ( 1 + np.exp( -1 * value ) )
        return sigmoid

    #Derivative of Sigmoid
    def derivative(self, value):
        derivative = value * ( 1 - value )
        return derivative

    #Calculate the sum of square error
    def calculate_loss(self, predicted, real):
        error_sum = 0
        for i in range( len(predicted) ):
            errorr = ( predicted[i] - real[i] ) ** 2
            error_sum += errorr
        return error_sum

    #Rearrange the weights
    def __add_weight(self, layer, amount):
        if ( layer == "input" ):
            self.__input_weight += amount
        elif ( layer == "hidden" ):
            self.__hidden_weight += amount
        else:
            self.__output_weight += amount
    
    #Train the network according to the train data
    def train(self, in_value, out_value, step, rate):
        index = 0
        for i in range( step ):
            
            if ( index == len(in_value) ):
                index = 0

            i_w = self.__input_weight
            h_w = self.__hidden_weight
            o_w = self.__output_weight

            val = i_w + h_w * in_value[index]

            act = self.sigmoid( val )
            der_act = self.derivative( act )
            
            predicted = np.sum(o_w * act)

            #Rearrange the weights according to the error
            err = out_value[index] - predicted
            in_amount = err * rate * o_w * der_act
            h_amount = in_amount * in_value[index]
            o_amount = err * rate * act
            
            self.__add_weight("input", in_amount)
            self.__add_weight("hidden", h_amount)
            self.__add_weight("output", o_amount)
            
            new_val = i_w + in_value.reshape(len(in_value), 1) * h_w
            act = self.sigmoid( new_val )
            predictions = np.dot( act, o_w )
            loss = self.calculate_loss(predictions, out_value)
            

            index += 1

    #Predict the data
    #Returns the total loss of prediction
    def predict(self, in_value, out_value):
        i_w = self.__input_weight
        h_w = self.__hidden_weight
        o_w = self.__output_weight

        val = i_w + in_value.reshape(len(in_value), 1) * h_w
        act = self.sigmoid( val )
        self.__predicted = np.dot( act, o_w )
        
        sum_loss = self.calculate_loss(self.__predicted, out_value)
        print("The total loss is " + str(sum_loss))
        return sum_loss

    #Plot the graph of prediction and actual value
    def plot_graph(self, in_value, out_value):
        sum_loss = self.calculate_loss(self.__predicted, out_value)
        plt.scatter( in_value, self.__predicted, label="Predicted")
        plt.scatter( in_value, out_value, label="Real")
        plt.xlabel("Input Values")
        plt.ylabel("Output Values")
        plt.legend()
        plt.title("ANN Data - Loss " + str(sum_loss))
        plt.show()