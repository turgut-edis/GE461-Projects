"""
In this question, Sammon's Mapping and t-SNE is used to map the digit data set to two dimensions.

In this code,
sklearn is used to implement t-SNE.
sammon is used to implement Sammon's Mapping. (It is taken from https://github.com/tompollard/sammon repository)
scipy.io is used to read the .mat file.
numpy is used to create arrays.

Author: Turgut Alp Edis
"""
#Importing necessary libraries
from sammon import sammon as SammonMapping
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load dataset
data = loadmat("./digits/digits.mat")

features = data["digits"]
labels = data["labels"]

"""Sammon Mapping Part"""
#Sammon implementation takes so much time.

#Implement Sammon Mapping for 50 iterations

[x,E] = SammonMapping(features, n=2, maxiter=50)

#Scatter the points for each digit
label_list = []
for i in range(10):
  label = str(i)
  plt.scatter(x[labels[:,0] == i, 0], x[labels[:,0] == i, 1], marker='.', label=label)
  label_list.append(label)

plt.title('Sammon Mapping for 50 iterations')
plt.legend(label_list, loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plt.show()

"""TSNE"""

#Implement t-SNE for various iterations with various perplexities (distance between the points)

#Initially for 1000 iterations with 30 perplexity
tSNE = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

label_list = []
for i in range(10):
  label = str(i)
  plt.scatter(x[labels[:,0] == i, 0], x[labels[:,0] == i, 1], marker='.', label=label)
  label_list.append(label)

plt.title('t-SNE Mapping for 1000 Iterations with 30 Perplexity')
plt.legend(label_list, loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plt.show()

#For 1000 iterations with 20 perplexity
tSNE = TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1000, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

label_list = []
for i in range(10):
  label = str(i)
  plt.scatter(x[labels[:,0] == i, 0], x[labels[:,0] == i, 1], marker='.', label=label)
  label_list.append(label)

plt.title('t-SNE Mapping for 1000 Iterations with 20 Perplexity')
plt.legend(label_list, loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plt.show()

#For 1500 iterations with 30 perplexity
tSNE = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1500, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

label_list = []
for i in range(10):
  label = str(i)
  plt.scatter(x[labels[:,0] == i, 0], x[labels[:,0] == i, 1], marker='.', label=label)
  label_list.append(label)
plt.title('t-SNE Mapping for 1500 Iterations with 30 Perplexity')
plt.legend(label_list, loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plt.show()

#For 1500 iterations with 20 perplexity
tSNE = TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0, learning_rate='warn', n_iter=1500, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

label_list = []
for i in range(10):
  label = str(i)
  plt.scatter(x[labels[:,0] == i, 0], x[labels[:,0] == i, 1], marker='.', label=label)
  label_list.append(label)

plt.title('t-SNE Mapping for 1500 Iterations with 20 Perplexity')
plt.legend(label_list, loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plt.show()

#For 2000 iterations with 30 perplexity
tSNE = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=2000, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

label_list = []
for i in range(10):
  label = str(i)
  plt.scatter(x[labels[:,0] == i, 0], x[labels[:,0] == i, 1], marker='.', label=label)
  label_list.append(label)

plt.title('t-SNE Mapping for 2000 Iterations with 30 Perplexity')
plt.legend(label_list, loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plt.show()

#For 2000 iterations with 20 perplexity
tSNE = TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0, learning_rate='warn', n_iter=2000, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

label_list = []
for i in range(10):
  label = str(i)
  plt.scatter(x[labels[:,0] == i, 0], x[labels[:,0] == i, 1], marker='.', label=label)
  label_list.append(label)
plt.title('t-SNE Mapping for 2000 Iterations with 20 Perplexity')
plt.legend(label_list, loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plt.show()

#For 4000 iterations with 30 perplexity
tSNE = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='warn', n_iter=4000, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

label_list = []
for i in range(10):
  label = str(i)
  plt.scatter(x[labels[:,0] == i, 0], x[labels[:,0] == i, 1], marker='.', label=label)
  label_list.append(label)
plt.title('t-SNE Mapping for 4000 Iterations with 30 Perplexity')
plt.legend(label_list, loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plt.show()

#For 4000 iterations with 20 perplexity
tSNE = TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0, learning_rate='warn', n_iter=4000, n_iter_without_progress=300, 
min_grad_norm=1e-07, metric='euclidean', init='warn', verbose=0, random_state=0, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')
x = tSNE.fit_transform(features)

label_list = []
for i in range(10):
  label = str(i)
  plt.scatter(x[labels[:,0] == i, 0], x[labels[:,0] == i, 1], marker='.', label=label)
  label_list.append(label)
plt.title('t-SNE Mapping for 4000 Iterations with 20 Perplexity')
plt.legend(label_list, loc='center left', bbox_to_anchor=(1, 0.5), title="Digits")
plt.show()

