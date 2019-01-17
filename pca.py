import pandas as pd
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler


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
traces = []
legend = {0: False, 1: False, 2: False, 3: True}
colors = {'Iris-setosa': 'rgb(31, 119, 180)',
          'Iris-versicolor': 'rgb(255, 127, 14)',
          'Iris-virginica': 'rgb(44, 160, 44)'}
for col in range(4):
    for key in colors:
        traces.append(Histogram(x=X[y == key, col], opacity=0.75,
                                xaxis='x%s' % (col+1), marker=Marker(color=colors[key]), name=key, showlegend=legend[col]))
data = Data(traces)
layout = Layout(barmode='overlay',
                xaxis=XAxis(domain=[0, 0.25],
                            title='sepal length (cm)'),
                xaxis2=XAxis(domain=[0.3, 0.5],
                             title='sepal width (cm)'),
                xaxis3=XAxis(domain=[0.55, 0.75],
                             title='petal length (cm)'),
                xaxis4=XAxis(domain=[0.8, 1],
                             title='petal width (cm)'),
                yaxis=YAxis(title='count'),
                title='Distribution of the different Iris flower features')

fig = Figure(data=data, layout=layout)
py.iplot(fig)


