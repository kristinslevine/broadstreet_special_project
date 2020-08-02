import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

m = pd.read_csv('asian_data.csv')
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',14)
print('\n')

#select columns to use for DT
m_tree = m[['State', 'Asian_June15', 'Reopening', 'Mask', 'over_100']]

#check to see if df has any NaN values
print(m_tree.isnull().sum())
print(m_tree.dtypes)

#enter variables as binary
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
m_tree['State']=le.fit_transform(m_tree['State'])
m_tree['Asian_June15']=le.fit_transform(m_tree['Asian_June15'])
m_tree['Reopening']=le.fit_transform(m_tree['Reopening'])
m_tree['Mask']=le.fit_transform(m_tree['Mask'])
m_tree['over_100']=le.fit_transform(m_tree['over_100'])


print(m_tree)

print('Decision Tree Algorithm')
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections
from sklearn.tree import DecisionTreeClassifier

# Data Collection
X = m_tree.values[:, 0:4]
Y = m_tree.values[:, 4]

data_feature_names = ['State', 'Asian_June15', 'Reopening', 'Mask']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

#Visualize data
dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('m_tree.png')
graph.write_svg('m_tree.svg')

# split the dataset
# separate the target variable
X = m_tree.values[:, 0:4]
y = m_tree.values[:, 4]

# encloding the class with sklearn's LabelEncoder
class_le = LabelEncoder()

# fit and transform the class
y = class_le.fit_transform(y)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

# performing training
clf_gini.fit(X_train, y_train)

# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
clf_entropy.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
# make predictions
# predicton on test using gini
y_pred_gini = clf_gini.predict(X_test)

# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)
#%%-----------------------------------------------------------------------
# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print ('-'*80 + '\n')
# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print ('-'*80 + '\n')
#%%-----------------------------------------------------------------------
# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = m_tree.over_100.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
class_names = m_tree.over_100.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()