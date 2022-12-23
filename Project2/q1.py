"""
In this question, principal components analysis (PCA) is used to project the 400-dimensional
data onto lower dimensional subspaces to observe the effect of dimensionality on the performance
of the Gaussian classifier.

In this code, 
sklearn is used to implement Gaussian classifier, calculate error statistics and split the data.
matplotlib is used to plot the data and draw the images.
scipy.io is used to read the .mat file.
numpy is used to compute necessary calculations for PCA and create arrays.

Author: Turgut Alp Edis
"""
#Importing necessary libraries
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
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
train_features, test_features, train_labels, test_labels = train_test_split (features, labels, test_size = 0.5, random_state = 0, shuffle = True)

"""Apply PCA Manually"""
#Initially, the used data is centered
train_center_X = train_features - np.mean(train_features)
#Then, the covariance matrix is calculated
cov_mat = np.cov(train_center_X, rowvar=False)
#From the covariance matrix, eigen values and eigen vectors are calculated
eigen_val, eigen_vec = np.linalg.eigh(cov_mat)
#Sorting the eigen values and vectors reversely
sorted_eigen = np.argsort(eigen_val)[::-1]
sorted_eigen_vec = eigen_vec[:, sorted_eigen]

#Calculating the sum of the eigen values to calculate explained variance ratio
total = eigen_val.sum()
#Calculating the PCs for this data
n_components = 400
pca = sorted_eigen_vec[:, 0:n_components]

#Calculating the explained variance and explained variance ratio
var_exp = eigen_val[sorted_eigen]
var_exp_ratio = var_exp / total

"""Question 1.1"""

#Plot the eigen values in descending order
plt.plot([x for x in range(1, 401)] , var_exp)
plt.title("Principal Components vs. Eigenvalues")
plt.xlabel("Principal Components")
plt.ylabel("Eigenvalues")
plt.show() 

"""Question 1.2"""

#Calculating the sample mean of the data
x_mean = np.mean(train_features, axis=0).reshape(20,20).T

# Display mean of data
plt.imshow(x_mean)
plt.title("Mean of Training Data")
plt.axis("off")
plt.show()

# Display mean of data as gray for better observation
plt.imshow(x_mean, cmap="gray")
plt.title("Mean of Training Data - Gray")
plt.axis("off")
plt.show()

#Display the eigen vectors

#To select eigen vectors, it is needed to calculate cumulative sum of explained variance ratio
print(var_exp_ratio.cumsum()) 
#According to testing, I choose 64 PCs since after 64, the difference between others are not large enough to be considered

row = 8 #Row number for plot table 
col = 8 #Column number for plot table

#Plot the PCs gray colored since the observation is easy when the image is gray
for i in range(row * col):
    plt.axis("off")
    plt.subplot(row, col, i+1)
    plt.imshow(pca[:,i].reshape(20,20).T, cmap="gray") #20x20
plt.suptitle("First 64 Components")
plt.show() 

test_stats = np.zeros((150,2))
train_stats = np.zeros((150,2))

"""Question 1.3"""

#The accuracy is calculated for first 50 dimension sizes
for i in range(150):
    idx = i + 1
    #Calculating the PCA
    pca = sorted_eigen_vec[:, 0:idx]
    train_center_X = train_features - np.mean(train_features)
    cov_mat_trainX = np.dot(train_center_X, pca[:,:])

    test_center_X = test_features - np.mean(test_features)
    cov_mat_testX = np.dot(test_center_X, pca[:,:])

    #Calculating the GaussianNB
    gaussian_classifier = GaussianNB()
    gaussian_classifier.fit(cov_mat_trainX, (train_labels.T)[0])

    #Predicting the train data
    train_pred = gaussian_classifier.predict(cov_mat_trainX)
    train_acc = metrics.accuracy_score((train_labels.T)[0], train_pred)
    train_err = 1 - train_acc

    #Predicting the test data
    test_pred = gaussian_classifier.predict(cov_mat_testX)
    test_acc = metrics.accuracy_score((test_labels.T)[0], test_pred)
    test_err = 1 - test_acc

    #Append the statistics to the list
    train_stats[i, 1] = train_err 
    train_stats[i, 0] = idx
    test_stats[i, 1] = test_err 
    test_stats[i, 0] = idx 

"""Question 1.4"""

#In the stat lists, 0 is used to hold the number of PC and 1 is used to hold the error
plt.plot(train_stats[:, 0], train_stats[:, 1], label="train")
plt.plot(test_stats[:, 0], test_stats[:, 1], label="test")
plt.title("Training Errors vs Test Errors")
plt.xlabel("Number of Components")
plt.ylabel("Classification Error")
plt.legend()
plt.show()

#Separating the lines into two plots
#Plot the train statistics
plt.plot(train_stats[:, 0], train_stats[:, 1])
plt.title("Training Result Statistic")
plt.xlabel("Number of Components")
plt.ylabel("Classification Error")
plt.show()

#Plot the test statistics
plt.plot(test_stats[:, 0], test_stats[:, 1])
plt.title("Test Result Statistic")
plt.xlabel("Number of Components")
plt.ylabel("Classification Error")
plt.show()
