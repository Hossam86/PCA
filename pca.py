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


