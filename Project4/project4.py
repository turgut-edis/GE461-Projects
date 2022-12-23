import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def k_means(dataset, K, iter_cnt, outliers=True):
    m = dataset.shape[0]  #Sample size
    n = dataset.shape[1]  #Feature size

    #Creating an empty centroid array to hold centroids
    centroids = np.array([]).reshape(n, 0)

    #Creating centroids with given k number
    for i in range(K):
        centroids = np.c_[centroids, dataset[random.randint(0, m-1)]]

    #Creating euclid array to hold euclidian distances
    euclid = np.array([]).reshape(m, 0)

    #Total Euclidian Distance for each centroid
    for i in range(K):
        dist = np.sum((dataset-centroids[:, i])**2, axis=1)
        euclid = np.c_[euclid, dist]

    #Storing the minimum distance
    minimum = np.argmin(euclid, axis=1)+1

    #Storing the clusters
    cent = {}
    for k in range(K):
        cent[k+1] = np.array([]).reshape(2, 0)

    #Each cluster to the point
    for k in range(m):
        cent[minimum[k]] = np.c_[cent[minimum[k]], dataset[k]]
    for k in range(K):
        cent[k+1] = cent[k+1].T
    
    #Computing the mean and update the centroid
    for k in range(K):
        centroids[:, k] = np.mean(cent[k+1], axis=0)

    #Creating the dictionary called final to store final positions of centroid
    #and group the points
    final = {}
    #Grouping continues for iteration count since it converges at some point
    for i in range(iter_cnt):
        euclid = np.array([]).reshape(m, 0)
        for k in range(K):
            dist = np.sum((dataset-centroids[:, k])**2, axis=1)
            euclid = np.c_[euclid, dist]
        C = np.argmin(euclid, axis=1)+1
        cent = {}
        for k in range(K):
            cent[k+1] = np.array([]).reshape(2, 0)
        for k in range(m):
            cent[C[k]] = np.c_[cent[C[k]], dataset[k]]
        for k in range(K):
            cent[k+1] = cent[k+1].T
        for k in range(K):
            centroids[:, k] = np.mean(cent[k+1], axis=0)
        final = cent

    #plot the graph
    for i in range(K):
        plt.scatter(final[i+1][:, 0], final[i+1][:, 1],
                    label=("Cluster " + str(i+1)))
    plt.legend()
    if outliers:
        plt.title("K Clustering for k=" + str(k+1))
    else:
        plt.title("K Clustering for k=" + str(k+1) + " without Outliers")
    plt.show()

    return final

'''Part A'''
dataset = pd.read_csv("falldetection_dataset.csv", header=None)

data_features = dataset.drop([0, 1], axis=1)
data_labels = dataset[1]

train_x_data = data_features.values
train_y_data = data_labels.values

"""Apply PCA Manually"""
# Initially, the used data is centered
train_center_X = train_x_data - np.mean(train_x_data)
# Then, the covariance matrix is calculated
cov_mat = np.cov(train_center_X, rowvar=False)
# From the covariance matrix, eigen values and eigen vectors are calculated
eigen_val, eigen_vec = np.linalg.eigh(cov_mat)
# Sorting the eigen values and vectors reversely
sorted_eigen = np.argsort(eigen_val)[::-1]
sorted_eigen_vec = eigen_vec[:, sorted_eigen]

# Calculating the sum of the eigen values to calculate explained variance ratio
total = eigen_val.sum()
# Calculating the PCs for this data
n_components = 2
pca = sorted_eigen_vec[:, 0:n_components]

# Calculating the explained variance and explained variance ratio
var_exp = eigen_val[sorted_eigen]
var_exp_ratio = var_exp / total

transformed_X = np.dot(pca.T, train_center_X.T).T
transformed_X = -1 * transformed_X

for idx in range(n_components):
    print(f"PC: {idx}, Variance: {var_exp_ratio[idx]}")
print("Total variance: ", sum(var_exp_ratio[:n_components]))

plt.scatter(transformed_X[train_y_data == 'F', 0],
            transformed_X[train_y_data == 'F', 1], label="Fall")
plt.scatter(transformed_X[train_y_data == 'NF', 0],
            transformed_X[train_y_data == 'NF', 1], label="Non-Fall")
plt.title('Data Projection PCA with 2 components')
plt.legend()
plt.show()

#K Means Clustering
for i in range(2, 9):
    k_means(transformed_X, i, 150)

#Removing outliers to see data more clearly
outlier_max = max(transformed_X[:, 1])
outlier_min = min(transformed_X[:, 1])

index_max = np.where((transformed_X[:, 1] == outlier_max) == True)[0][0]
index_min = np.where((transformed_X[:, 1] == outlier_min) == True)[0][0]

transformed_X_wo_ = np.delete(transformed_X, [index_max, index_min], axis=0)
train_y_wo_ = np.delete(train_y_data, [index_max, index_min], axis=0)

for i in range(2, 9):
    k_means(transformed_X_wo_, i, 150, False)

k_means_2 = k_means( transformed_X, 2, 150, False )

print( transformed_X[train_y_data == 'F'].shape, k_means_2[1].shape, transformed_X[train_y_data == 'NF'].shape, k_means_2[2].shape )
consistency_1 = 0
if k_means_2[1].shape[0] > k_means_2[2].shape[0]:
    if k_means_2[1].shape[0] > transformed_X[train_y_data == 'F'].shape[0]:
        for i in range ( transformed_X[train_y_data == 'F'].shape[0] ):
            if k_means_2[1][i,0] == transformed_X[train_y_data == 'F'][i,0] and  k_means_2[1][i,1] == transformed_X[train_y_data == 'F'][i,1]:
                consistency_1 += 1

print(consistency_1)

plt.scatter(transformed_X_wo_[train_y_wo_ == 'F', 0],
            transformed_X_wo_[train_y_wo_ == 'F', 1], label="Fall")
plt.scatter(transformed_X_wo_[train_y_wo_ == 'NF', 0],
            transformed_X_wo_[train_y_wo_ == 'NF', 1], label="Non-Fall")
plt.title('Data Projection PCA with 2 components without outliers')
plt.legend()
plt.show()

'''Part B'''

#Splitting the Dataset into 70% train, 15% test and 15% validation
train__x, test__x, train__y, test__y = train_test_split(train_x_data, train_y_data, test_size=0.30)
valid__x, test__x, valid__y, test__y = train_test_split(test__x, test__y, test_size=0.50)

#Values of parameters for SVM
reg_param = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
degree = [1, 2, 3, 4, 5]
kernel = ["linear", "poly", "rbf", "sigmoid"]

svm_results = []
for reg in reg_param:
    for k in kernel:
        #First for linear kernel with regularization parameter = 1e-3
        if k == "linear":
            svm_model = SVC( C = reg, kernel = k, max_iter = 1000 )
            svm_model.fit(train__x, train__y)
            prediction = svm_model.predict(valid__x)
            acc = metrics.accuracy_score(valid__y, prediction)

            data = {'Regularization Parameter': reg,
                    'Kernel Type': k,
                    'Degree': "",
                    'Validation Accuracy': acc}

            svm_results.append(data)
            print(data)
        #Then poly kernel
        elif k == "poly":
            #Check poly in multiple degrees
            for d in degree:
                svm_model = SVC( C = reg, kernel=k, degree=d, max_iter=10000 )
                svm_model.fit(train__x, train__y)
                prediction = svm_model.predict(valid__x)
                acc = metrics.accuracy_score(valid__y, prediction)

                data = {'Regularization Parameter': reg,
                        'Kernel Type': k,
                        'Degree': degree,
                        'Validation Accuracy': acc}

                svm_results.append(data)
                print(data)
        else:
            #Check other kernels
            svm_model = SVC( C = reg, kernel = k, max_iter = 10000 )
            svm_model.fit(train__x, train__y)
            prediction = svm_model.predict(valid__x)
            acc = metrics.accuracy_score(valid__y, prediction)

            data = {'Regularization Parameter': reg,
                    'Kernel Type': k,
                    'Degree': "",
                    'Validation Accuracy': acc}
            svm_results.append(data)
            print(data)

#Sort the list of results according to validation accuracy
sorted_SVM = sorted( svm_results, key = lambda x: x['Validation Accuracy'], reverse=True)
SVM_data = pd.DataFrame(sorted_SVM)

#Writing the results to the xlsx file to make easy include to the report
writer = pd.ExcelWriter('DataSVM.xlsx')
SVM_data.to_excel(writer, 'Sheet1', index=False)
writer.save()
print('SVM_data is written to Excel File successfully.')

#Showing best SVM Result
best_SVM = sorted_SVM[0]
reg = best_SVM['Regularization Parameter']
k = best_SVM['Kernel Type']
d = best_SVM['Degree']
acc = best_SVM['Validation Accuracy']

print("--- Best SVM model is ---")
print("Regularization Parameter:", reg)
print("Kernel Type:", k)
print("Degree:", d)
print("Validation Accuracy:", acc)

if d != "":
    best_SVM = SVC( C = reg, kernel = k, degree = d, max_iter = 10000 )
    best_SVM.fit(train__x, train__y)
    prediction = best_SVM.predict(test__x)
    acc = metrics.accuracy_score(test__y, prediction)
    print("Accuracy of the best SVM: ", acc)

else:
    best_SVM = SVC( C = reg, kernel = k, max_iter = 10000 )
    best_SVM.fit(train__x, train__y)
    prediction = best_SVM.predict(test__x)
    acc = metrics.accuracy_score(test__y, prediction)
    print("Accuracy of the best SVM: ", acc)

#Calculating MLP
#Parameters
reg = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
hidden_layer = [(8, 8), (16, 16), (32, 32), (64, 64)]
activate = ["logistic", "tanh", "relu"]
l_rates = [1e-3, 1e-2, 1e-1]

mlp_results = []

#Trying each combination of parameters
for layer_size in hidden_layer:

    for activation in activate:

        for r in reg:

            for learning in l_rates:

                mlp_model = MLPClassifier(hidden_layer_sizes=layer_size, activation=activation, alpha=r, learning_rate_init=learning, max_iter=5000)
                mlp_model.fit(train__x, train__y)
                prediction = mlp_model.predict(valid__x)
                acc = metrics.accuracy_score(valid__y, prediction)

                data = {"Hidden Layer Size": layer_size,
                        "Activation Function": activation,
                        "Regularization Size (Alpha)": r,
                        "Learning Rate": learning,
                        "Validation Accuracy": acc}
                
                mlp_results.append(data)
                print(data)

sorted_MLP = sorted(mlp_results, key=lambda x: x['Validation Accuracy'], reverse=True)
MLP_data = pd.DataFrame(sorted_MLP)

#Saving to the xlsx file to make easy to include to the report
writer = pd.ExcelWriter('DataMLP.xlsx')
MLP_data.to_excel(writer, 'Sheet1', index=False)
writer.save()
print('MLP_data is written to Excel File successfully.')

best_MLP = sorted_MLP[18]
hidden_layer = best_MLP['Hidden Layer Size']
activation = best_MLP['Activation Function']
reg = best_MLP['Regularization Size (Alpha)']
l_rate = best_MLP['Learning Rate']
val_acc = best_MLP['Validation Accuracy']

print("Best MLP model:")
print("Hidden Layer Size: ", hidden_layer)
print("Activation Function: ", activation)
print("Regularization Size (Alpha):", reg)
print("Learning Rate: ", l_rate)
print("Validation Accuracy: ", val_acc)

best_MLP = MLPClassifier( hidden_layer_sizes = hidden_layer, activation = activation, alpha = reg, learning_rate_init = l_rate, max_iter = 2000 )
best_MLP.fit(train__x, train__y)
prediction = best_MLP.predict(test__x)
acc = metrics.accuracy_score(test__y, prediction)
print("Accuracy of the best MLP: ", acc)