"""
In this question, Fisher linear discriminant analysis (LDA) is used to project the 400-dimensional
data onto lower dimensional subspaces.

In this code, 
sklearn is used to implement Linear Discriminant Analysis (LDA),
implement Gaussian classifier, calculate error statistics and split the data.
matplotlib is used to plot the data and draw the images.
scipy.io is used to read the .mat file.
numpy is used to create arrays.

Author: Turgut Alp Edis
"""
#Importing necessary libraries
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

"""Load the Dataset and split it for the train and test
As indicated in the document, 50 percent of the data is used for training and rest is used for testing"""

# Load dataset
data = loadmat("./digits/digits.mat")

features = data["digits"]
labels = data["labels"]
#Convert data to numpy array to ensure that it is fine to be used for calculations
features = np.array(features)
labels = np.array(labels)

# Split data into half between train and test
train_X, test_X, train_Y, test_Y = train_test_split(features, labels, test_size=0.5, random_state=0, shuffle=True)

#Transpose the label data to apply LDA
transposed_y = train_Y.T
transposed_test_y = test_Y.T

#Apply LDA
lda = LDA(n_components=9)
lda.fit(train_X, transposed_y[0])

scls = lda.scalings_
dimension = 9

row = 2
col = 5

"""Question 2.1"""
for i in range(dimension):
    plt.axis("off")
    plt.subplot(row, col, i+1)
    plt.imshow((scls[:, i]).reshape(20, 20))
plt.suptitle("Bases")
plt.show()

components = 9
test_stats = np.zeros((9,2))
train_stats = np.zeros((9,2))

#The accuracy is calculated for 9 dimensions
for i in range(components):
    idx = i + 1
    #Calculating the LDA
    lda = LDA(n_components=idx)
    lda.fit(train_X, transposed_y[0])
    transformed_train_X = lda.transform(train_X)
    transformed_test_X = lda.transform(test_X)
    

    #Calculating the GaussianNB
    gaussian_classifier = GaussianNB()
    gaussian_classifier.fit(transformed_train_X, (transposed_y)[0])

    #Predicting the train data
    train_pred = gaussian_classifier.predict(transformed_train_X)
    train_acc = metrics.accuracy_score((transposed_y)[0], train_pred)
    train_err = 1 - train_acc

    #Predicting the test data
    test_pred = gaussian_classifier.predict(transformed_test_X)
    test_acc = metrics.accuracy_score((transposed_test_y)[0], test_pred)
    test_err = 1 - test_acc

    #Append the statistics to the list
    train_stats[i, 1] = train_err 
    train_stats[i, 0] = idx
    test_stats[i, 1] = test_err 
    test_stats[i, 0] = idx 

"""Question 2.3"""

#In the stat lists, 0 is used to hold the number of PC and 1 is used to hold the error

#Plot the train statistics
plt.plot(train_stats[:, 0], train_stats[:, 1])
plt.title("Training Result Statistic")
plt.xlabel("Number of Dimensions")
plt.ylabel("Classification Error")
plt.show()

#Plot the test statistics
plt.plot(test_stats[:, 0], test_stats[:, 1])
plt.title("Test Result Statistic")
plt.xlabel("Number of Dimensions")
plt.ylabel("Classification Error")
plt.show()