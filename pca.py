import pandas as pd
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler
import numpy as np


# Loading the Dataset
df = pd.read_csv(
    filepath_or_buffer="/media/hossam/MyFiles/MachineLearning/PCA/PCA/IrisDataSet/iris.data", header=None, sep=',')
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True)  # drops the empty line at file-end
print(df.head())


# split data table into data X and class labels y
X = df.ix[:, 0:4].values
y = df.ix[:, 4]

# Our iris dataset is now stored in form of a 150 x 4 matrix where the columns are the different features,
#  and every row represents a separate flower sample. Each sample row  can be pictured as a 4-dimensional vector

# Exploratory Visualization
# plotting histograms
# traces = []
# legend = {0: False, 1: False, 2: False, 3: True}
# colors = {'Iris-setosa': 'rgb(31, 119, 180)',
#           'Iris-versicolor': 'rgb(255, 127, 14)',
#           'Iris-virginica': 'rgb(44, 160, 44)'}
# for col in range(4):
#     for key in colors:
#         traces.append(Histogram(x=X[y == key, col], opacity=0.75,
#                                 xaxis='x%s' % (col+1), marker=Marker(color=colors[key]), name=key, showlegend=legend[col]))
# data = Data(traces)
# layout = Layout(barmode='overlay',
#                 xaxis=XAxis(domain=[0, 0.25],
#                             title='sepal length (cm)'),
#                 xaxis2=XAxis(domain=[0.3, 0.5],
#                              title='sepal width (cm)'),
#                 xaxis3=XAxis(domain=[0.55, 0.75],
#                              title='petal length (cm)'),
#                 xaxis4=XAxis(domain=[0.8, 1],
#                              title='petal width (cm)'),
#                 yaxis=YAxis(title='count'),
#                 title='Distribution of the different Iris flower features')

# fig = Figure(data=data, layout=layout)
# py.iplot(fig)

# Standardizing
# Whether to standardize the data prior to a PCA on the covariance matrix depends on
# the measurement scales of the original features. Since PCA yields a feature subspace
# that maximizes the variance along the axes, it makes sense to standardize the data,
# especially, if it was measured on different scales. Although, all features in the Iris
# dataset were measured in centimeters, let us continue with the transformation of
# the data onto unit scale (mean=0 and variance=1), which is a requirement for the optimal
# performance of many machine learning algorithms.

X_std = StandardScaler().fit_transform(X)

# The eigenvectors and eigenvalues of a covariance (or correlation) matrix represent the "core" of a PCA:
# The eigenvectors (principal components) determine the directions of the new feature space, and the eigenvalues determine their magnitude.
# In other words, the eigenvalues explain the variance of the data along the new feature axes.

# Covariance Matrix
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std-mean_vec).T.dot((X_std-mean_vec))/(X_std.shape[0]-1)
print('Covariance matrix \n%s' % cov_mat)

# built in Cov function
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' % eig_vecs)
print('\nEigenvalues \n%s' % eig_vals)
