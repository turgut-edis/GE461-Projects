from matplotlib import pyplot as plt
import numpy as np

"""
This class calculates the Linear Regression of given 1-D input and output
Author: Turgut Alp Edis
"""
class LinReg:
    #Initialize the linear regression with giving the input and output data
    def __init__(self, in_value, out_value):
        self.__input = in_value
        self.__output = out_value
        self.__length = len(self.__input)
    
    #Calculate the sum of square error
    def calculate_loss(self, predicted, real):
        error_sum = 0
        for i in range(len(predicted)):
            errorr = (predicted[i] - real[i]) ** 2
            error_sum += errorr
        return error_sum

    #Calculate the regression with common formula
    def calculate_regression(self):
        self.__mean_x = np.mean(self.__input)
        self.__mean_y = np.mean(self.__output)

        dev_xy = 0
        for i in range(self.__length):
            productt = self.__input[i] * self.__output[i]
            dev_xy += productt
        dev_xy -= self.__length * self.__mean_x * self.__mean_y

        dev_xx = 0
        for i in range(self.__length):
            productt = self.__input[i] * self.__input[i]
            dev_xx += productt
        dev_xx -= self.__length * self.__mean_x * self.__mean_x

        self.__b_1 = dev_xy / dev_xx
        self.__b_0 = self.__mean_y - self.__b_1 * self.__mean_x

    #Plot the regression line
    def plot_line(self):
        plt.scatter(self.__input, self.__output, label="Real")

        predicted = self.__b_0 + self.__b_1 * self.__input
        loss = self.calculate_loss(predicted, self.__output)
        plt.plot(self.__input, predicted, label="Predicted")

        plt.xlabel('input')
        plt.ylabel('output')
        plt.legend()
        print("Loss ", loss)
        plt.title("Linear Regression Loss: " + str(loss))
        plt.show()