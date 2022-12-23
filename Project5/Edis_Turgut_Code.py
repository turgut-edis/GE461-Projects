# %%
#Import necessary packages to generate the dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from skmultiflow.data import HyperplaneGenerator

from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.lazy import KNNClassifier
from sklearn.naive_bayes import GaussianNB

from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream

from sklearn.ensemble import VotingClassifier

import numpy as np
import pandas as pd

# %%
#Generate the Dataset

#a) Hyperplane Dataset (noise= 10%, number of drifting features 2)
hyper_dataset_generated = HyperplaneGenerator(n_features=10, n_drift_features=2, noise_percentage=0.1)
hyper_dataset_a = hyper_dataset_generated.next_sample(20000)
hyper_dataset_a = np.append(hyper_dataset_a[0], np.reshape(hyper_dataset_a[1], (20000,1)), axis=1)
hyper_dataset_a_df = pd.DataFrame(hyper_dataset_a)
hyper_dataset_a_df = hyper_dataset_a_df.astype({10: int})
hyper_dataset_a_df.to_csv("Hyperplane Dataset 10_2.csv", index=False)

hyper_dataset_generated = HyperplaneGenerator(n_features=10, n_drift_features=2, noise_percentage=0.3)
hyper_dataset_b = hyper_dataset_generated.next_sample(20000)
hyper_dataset_b = np.append(hyper_dataset_b[0], np.reshape(hyper_dataset_b[1], (20000,1)), axis=1)
hyper_dataset_b_df = pd.DataFrame(hyper_dataset_b)
hyper_dataset_b_df = hyper_dataset_b_df.astype({10: int})
hyper_dataset_b_df.to_csv("Hyperplane Dataset 30_2.csv", index=False)

hyper_dataset_generated = HyperplaneGenerator(n_features=10, n_drift_features=5, noise_percentage=0.1)
hyper_dataset_c = hyper_dataset_generated.next_sample(20000)
hyper_dataset_c = np.append(hyper_dataset_c[0], np.reshape(hyper_dataset_c[1], (20000,1)), axis=1)
hyper_dataset_c_df = pd.DataFrame(hyper_dataset_c)
hyper_dataset_c_df = hyper_dataset_c_df.astype({10: int})
hyper_dataset_c_df.to_csv("Hyperplane Dataset 10_5.csv", index=False)

hyper_dataset_generated = HyperplaneGenerator(n_features=10, n_drift_features=5, noise_percentage=0.3)
hyper_dataset_d = hyper_dataset_generated.next_sample(20000)
hyper_dataset_d = np.append(hyper_dataset_d[0], np.reshape(hyper_dataset_d[1], (20000,1)), axis=1)
hyper_dataset_d_df = pd.DataFrame(hyper_dataset_d)
hyper_dataset_d_df = hyper_dataset_d_df.astype({10: int})
hyper_dataset_d_df.to_csv("Hyperplane Dataset 30_5.csv", index=False)


# %%
#Training and Evaluating the Datasets, with using ht, knn and nb online learners

hyper_dataset_read_10_2 = FileStream("Hyperplane Dataset 10_2.csv")
hyper_dataset_read_30_2 = FileStream("Hyperplane Dataset 30_2.csv")
hyper_dataset_read_10_5 = FileStream("Hyperplane Dataset 10_5.csv")
hyper_dataset_read_30_5 = FileStream("Hyperplane Dataset 30_5.csv")

#Creating the classifiers

HT_classifier = HoeffdingTreeClassifier()
KNN_classifier = KNNClassifier()
NB_classifier = GaussianNB()

evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"])

evaluator.evaluate(stream=hyper_dataset_read_10_2, model=[HT_classifier, KNN_classifier, NB_classifier], model_names=["HT", "KNN", "NB"])
evaluator.evaluate(stream=hyper_dataset_read_30_2, model=[HT_classifier, KNN_classifier, NB_classifier], model_names=["HT", "KNN", "NB"])
evaluator.evaluate(stream=hyper_dataset_read_10_5, model=[HT_classifier, KNN_classifier, NB_classifier], model_names=["HT", "KNN", "NB"])
evaluator.evaluate(stream=hyper_dataset_read_30_5, model=[HT_classifier, KNN_classifier, NB_classifier], model_names=["HT", "KNN", "NB"])

# %%
#Different Batch Sizes

for size in [1,100,1000]:
    HT_online_classifier = HoeffdingTreeClassifier()
    KNN_online_classifier = KNNClassifier()
    NB_classifier = GaussianNB()

    print("Evaluating for batch size=", size, " and dataset Hyperplane Dataset")
    evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"], batch_size=size)
    evaluator.evaluate(stream=hyper_dataset_read_30_2, model=[HT_online_classifier, KNN_online_classifier, NB_classifier], model_names=["HT", "KNN", "NB"])

for size in [1,100,1000]:
    HT_online_classifier = HoeffdingTreeClassifier()
    KNN_online_classifier = KNNClassifier()
    NB_classifier = GaussianNB()

    print("Evaluating for batch size=", size, " and dataset Hyperplane Dataset")
    evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"], batch_size=size)
    evaluator.evaluate(stream=hyper_dataset_read_30_2, model=[HT_online_classifier, KNN_online_classifier, NB_classifier], model_names=["HT", "KNN", "NB"])

for size in [1,100,1000]:
    HT_online_classifier = HoeffdingTreeClassifier()
    KNN_online_classifier = KNNClassifier()
    NB_classifier = GaussianNB()

    print("Evaluating for batch size=", size, " and dataset Hyperplane Dataset")
    evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"], batch_size=size)
    evaluator.evaluate(stream=hyper_dataset_read_10_5, model=[HT_online_classifier, KNN_online_classifier, NB_classifier], model_names=["HT", "KNN", "NB"])

for size in [1,100,1000]:
    HT_online_classifier = HoeffdingTreeClassifier()
    KNN_online_classifier = KNNClassifier()
    NB_classifier = GaussianNB()

    print("Evaluating for batch size=", size, " and dataset Hyperplane Dataset")
    evaluator = EvaluatePrequential(show_plot=True, metrics=["accuracy"], batch_size=size)
    evaluator.evaluate(stream=hyper_dataset_read_30_5, model=[HT_online_classifier, KNN_online_classifier, NB_classifier], model_names=["HT", "KNN", "NB"])



# %%
hyper_dataset_10_2 = pd.read_csv("Hyperplane Dataset 10_2.csv").values
hyper_dataset_10_2_features = hyper_dataset_10_2[:,:10]
hyper_dataset_10_2_labels = np.array(hyper_dataset_10_2[:,10], dtype=int)
train_features_10_2, test_features_10_2, train_labels_10_2, test_labels_10_2 = train_test_split(hyper_dataset_10_2_features, hyper_dataset_10_2_labels, test_size=0.3)

hyper_dataset_30_2 = pd.read_csv("Hyperplane Dataset 30_2.csv").values
hyper_dataset_features_30_2 = hyper_dataset_30_2[:,:10]
hyper_dataset_labels_30_2 = np.array(hyper_dataset_30_2[:,10], dtype=int)
train_features_30_2, test_features_30_2, train_labels_30_2, test_labels_30_2 = train_test_split(hyper_dataset_features_30_2, hyper_dataset_labels_30_2, test_size=0.3)

hyper_dataset_10_5 = pd.read_csv("Hyperplane Dataset 10_5.csv").values
hyper_dataset_features_10_5 = hyper_dataset_10_5[:,:10]
hyper_dataset_labels_10_5 = np.array(hyper_dataset_10_5[:,10], dtype=int)
train_features_10_5, test_features_10_5, train_labels_10_5, test_labels_10_5 = train_test_split(hyper_dataset_features_10_5, hyper_dataset_labels_10_5, test_size=0.3)

hyper_dataset_30_5 = pd.read_csv("Hyperplane Dataset 30_5.csv").values
hyper_dataset_features_30_5 = hyper_dataset_30_5[:,:10]
hyper_dataset_labels_30_5 = np.array(hyper_dataset_30_5[:,10], dtype=int)
train_features_30_5, test_features_30_5, train_labels_30_5, test_labels_30_5 = train_test_split(hyper_dataset_features_30_5, hyper_dataset_labels_30_5, test_size=0.3)

# %%
preds = {}
HT_classifier = HoeffdingTreeClassifier()
KNN_classifier = KNNClassifier()
NB_classifier = GaussianNB()

HT_classifier.fit(train_features_10_2, train_labels_10_2)
pred_ = HT_classifier.predict(test_features_10_2)
acc = accuracy_score(test_labels_10_2, pred_)
preds[0] = pred_

KNN_classifier.fit(train_features_10_2, train_labels_10_2)
pred_ = KNN_classifier.predict(test_features_10_2)
acc = accuracy_score(test_labels_10_2, pred_)
preds[1] = pred_

NB_classifier.fit(train_features_10_2, train_labels_10_2)
pred_ = NB_classifier.predict(test_features_10_2)
acc = accuracy_score(test_labels_10_2, pred_)
preds[2] = pred_

HT_classifier.fit(train_features_30_2, train_labels_30_2)
pred_ = HT_classifier.predict(test_features_30_2)
acc = accuracy_score(test_labels_30_2, pred_)
preds[3] = pred_

KNN_classifier.fit(train_features_30_2, train_labels_30_2)
pred_ = KNN_classifier.predict(test_features_30_2)
acc = accuracy_score(test_labels_30_2, pred_)
preds[4] = pred_

NB_classifier.fit(train_features_30_2, train_labels_30_2)
pred_ = NB_classifier.predict(test_features_30_2)
acc = accuracy_score(test_labels_30_2, pred_)
preds[5] = pred_

HT_classifier.fit(train_features_10_5, train_labels_10_5)
pred_ = HT_classifier.predict(test_features_10_5)
acc = accuracy_score(test_labels_10_5, pred_)
preds[6] = pred_

KNN_classifier.fit(train_features_10_5, train_labels_10_5)
pred_ = KNN_classifier.predict(test_features_10_5)
acc = accuracy_score(test_labels_10_5, pred_)
print(acc)
preds[7] = pred_

NB_classifier.fit(train_features_10_5, train_labels_10_5)
pred_ = NB_classifier.predict(test_features_10_5)
acc = accuracy_score(test_labels_10_5, pred_)
print(acc)
preds[8] = pred_

HT_classifier.fit(train_features_30_5, train_labels_30_5)
pred_ = HT_classifier.predict(test_features_30_5)
acc = accuracy_score(test_labels_30_5, pred_)
print(acc)
preds[9] = pred_

KNN_classifier.fit(train_features_30_5, train_labels_30_5)
pred_ = KNN_classifier.predict(test_features_30_5)
acc = accuracy_score(test_labels_30_5, pred_)
print(acc)
preds[10] = pred_

NB_classifier.fit(train_features_30_5, train_labels_30_5)
pred_ = NB_classifier.predict(test_features_30_5)
acc = accuracy_score(test_labels_30_5, pred_)
print(acc)
preds[11] = pred_

ht = preds[0]
knn = preds[1]
nb = preds[2]
new_predict = np.zeros(len(ht))
for j in range(len(ht)):
   num = ht[j] + knn[j] + nb[j]
   num /= 3
   if num < 1:
        new_predict[j] = 0
   else:
       new_predict[j] = 1
acc = accuracy_score(test_labels_10_2, new_predict)
print(acc)

ht = preds[3]
knn = preds[4]
nb = preds[5]
new_predict = np.zeros(len(ht))
for j in range(len(ht)):
   num = ht[j] + knn[j] + nb[j]
   num /= 3
   if num < 1:
        new_predict[j] = 0
   else:
       new_predict[j] = 1
acc = accuracy_score(test_labels_30_2, new_predict)
print(acc)

ht = preds[6]
knn = preds[7]
nb = preds[8]
new_predict = np.zeros(len(ht))
for j in range(len(ht)):
   num = ht[j] + knn[j] + nb[j]
   num /= 3
   if num < 1:
        new_predict[j] = 0
   else:
       new_predict[j] = 1
acc = accuracy_score(test_labels_10_5, new_predict)
print(acc)

ht = preds[9]
knn = preds[10]
nb = preds[11]
new_predict = np.zeros(len(ht))
for j in range(len(ht)):
   num = ht[j] + knn[j] + nb[j]
   num /= 3
   if num < 1:
        new_predict[j] = 0
   else:
       new_predict[j] = 1
acc = accuracy_score(test_labels_30_5, new_predict)
print(acc)



