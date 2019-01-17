import pandas as pd
# import plotly.plotly as py
# rom plotly.graph_objs import *
# import plotly.tools as tls
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

# Correlation Matrix
# Especially, in the field of "Finance," the correlation matrix typically used instead of the
# covariance matrix. However, the eigendecomposition of the covariance matrix
# (if the input data was standardized) yields the same results as a eigendecomposition on
# the correlation matrix, since the correlation matrix can be understood as the normalized
#  covariance matrix. Eigendecomposition of the standardized data based on the correlation matrix:

cor_mat1 = np.corrcoef(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

print('Eigenvectors \n%s' % eig_vecs)
print('\nEigenvalues \n%s' % eig_vals)

# Eigendecomposition of the raw data based on the correlation matrix:
cor_mat2 = np.corrcoef(X.T)
eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('Eigenvectors \n%s' % eig_vecs)
print('\nEigenvalues \n%s' % eig_vals)

# We can clearly see that all three approaches yield the same eigenvectors and eigenvalue pairs:

# Eigendecomposition of the covariance matrix after standardizing the data.
# Eigendecomposition of the correlation matrix.
# Eigendecomposition of the correlation matrix after standardizing the data.
# ============================================================================================
# Singular Vector Decomposition
# ==============================
# While the eigendecomposition of the covariance or correlation matrix may be more intuitiuve,
# most PCA implementations perform a Singular Vector Decomposition (SVD) to improve
# the computational efficiency. So, let us perform an SVD to confirm that the result
# are indeed the same:

u, s, v = np.linalg.svd(X_std.T)

# Selecting Principal Components
# ================================
# The typical goal of a PCA is to reduce the dimensionality of the original feature space
#  by projecting it onto a smaller subspace, where the eigenvectors will form the axes. However,
#  the eigenvectors only define the directions of the new axis, since they have all the same
#  unit length 1, which can confirmed by the following two lines of code:

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# In order to decide which eigenvector(s) can dropped without losing too much information
# for the construction of lower-dimensional subspace, we need to inspect the corresponding eigenvalues:
# The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data;
# those are the ones can be dropped.In order to do so, the common approach is to rank the
# eigenvalues from highest to lowest in order choose the top k eigenvectors.

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
             for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

# After sorting the eigenpairs, the next question is "how many principal components are
# we going to choose for our new feature subspace?"
# A useful measure is the so-called "explained variance," which can be calculated from
# the eigenvalues. The explained variance tells us how much information (variance)
# can be attributed to each of the principal components.

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# trace1 = Bar(
#         x=['PC %s' %i for i in range(1,5)],
#         y=var_exp,
#         showlegend=False)

# trace2 = Scatter(
#         x=['PC %s' %i for i in range(1,5)],
#         y=cum_var_exp,
#         name='cumulative explained variance')

# data = Data([trace1, trace2])

# layout=Layout(
#         yaxis=YAxis(title='Explained variance in percent'),
#         title='Explained variance by different principal components')

# fig = Figure(data=data, layout=layout)
# py.iplot(fig)

# The plot above clearly shows that most of the variance (72.77% of the variance to be precise)
# can be explained by the first principal component alone. The second principal component still
# bears some information (23.03%) while the third and fourth principal components can safely be
# dropped without losing to much information. Together, the first two principal components contain
# 95.8% of the information.
#
# The construction of the projection matrix that will be used to transform the Iris data onto the
# new feature subspace. Although, the name "projection matrix" has a nice ring to it, it is basically
# just a matrix of our concatenated top k eigenvectors.
#
# Here, we are reducing the 4-dimensional feature space to a 2-dimensional feature subspace,
#  by choosing the "top 2" eigenvectors with the highest eigenvalues to construct
#  our d×k-dimensional eigenvector matrix W.


matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),
                      eig_pairs[1][1].reshape(4, 1)))

print('Matrix W:\n', matrix_w)
# =============================================
# Projection Onto the New Feature Space
# ===============================================
# In this last step we will use the 4×2-dimensional projection matrix W to transform our samples
# onto the new subspace via the equation Y=X×W, where Y is a 150×2 matrix of our transformed samples.

Y = X_std.dot(matrix_w)

# # plots
# traces = []
# trace = Scatter(x=Y[y == name, 0], y=Y[y == name, 1], mode='markers', name=name, marker=Marker(
#     size=12, line=Line(color='rgba(217, 217, 217, 0.14)', width=0.5), opacity=0.8))

# traces.append(trace)


# data = Data(traces)
# layout = Layout(showlegend=True, scene=Scene(
#     xaxis=XAxis(title='PC1'), yaxis=YAxis(title='PC2'),))

# fig = Figure(data=data, layout=layout)
# py.iplot(fig)
