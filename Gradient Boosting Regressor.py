# Fitting a data by using Gradient Boosting Regressor
# to determine 5 best and worst factors (by importance)
# finally getting tree's rule in a text
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# load data
data=pd.read_csv('_data.csv', sep=';')

X = data.drop(['lot_size'],axis=1)
X = X.drop(['Output'],axis=1)
X = X.drop(['t_minute'],axis=1)
X = X.drop(['score'],axis=1)
y = data['score']


factors = X.columns.values.tolist()

gb = GradientBoostingRegressor(n_estimators=100)
gb.fit(X, y)

_list=gb.feature_importances_
mylist=list(range(1,len(_list)+1))

# initialize data of lists.
data1 = {'factor_name': factors,
        'importance': _list}

# Create DataFrame
df = pd.DataFrame(data1)
df1=df.sort_values(by='importance', ascending=False)[:3]
df2=df.sort_values(by='importance')[:3]

class Importance: 
    def importance_factors(self):
        self.fig = px.bar(df, x="factor_name", y="importance",
                      barmode="overlay", color="factor_name",
                     orientation="v", color_discrete_sequence=[
                         px.colors.qualitative.Alphabet[6],
                         px.colors.qualitative.Alphabet[11],
                       px.colors.qualitative.Plotly[2],
                         px.colors.qualitative.Plotly[7],
                       px.colors.qualitative.G10[5]],
                     title="Importance Levels of All Factors")


        self.fig.show()
        
    def correlation(self):
        self.model = LogisticRegression(solver='liblinear', random_state=0)
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        return self.coef_

    # Best 5 factors
    def best_factors(self):
        self.fig = px.bar(df1, x="factor_name", y="importance",
                      barmode='overlay', color="factor_name",
                     orientation="v", hover_name="factor_name",
                     color_discrete_sequence=[
                         "orange", "red", "green"],
                     title="Most important factors leading to good outcomes")
        self.fig.show()
        
    def best_factors_direction(self):
        self.listt = []
        self.coef_=self.correlation()
        a=df1[0:5].index
    
        for i in a:
            if self.coef_[0][i] <0:
                self.listt.append("red")
            else:
                self.listt.append("blue")

        self.fig1 = px.bar(df1, x="factor_name", y="importance",
                      barmode='overlay', color="factor_name",
                     orientation="v", hover_name="factor_name",
                     color_discrete_sequence=self.listt,
                     title="Direction of the Most important factors")
        

        self.fig1.show()

    # Worst 5 factors
    def worst_factors(self):   
        self.fig = px.bar(df2, x="factor_name", y="importance",
                      barmode='overlay', color="factor_name",
                     orientation="v", hover_name="factor_name",
                     color_discrete_sequence=[
                         "purple", "blue","green"],
                     title="Least important factors leading to good outcomes")

        self.fig.show()
        
    def worst_factors_direction(self):
        self.listt = []
        self.coef_=self.correlation()
        a=df2[0:5].index
    
        for i in a:
            if self.coef_[0][i] <0:
                self.listt.append("red")
            else:
                self.listt.append("blue")

        self.fig = px.bar(df2, x="factor_name", y="importance",
                      barmode='overlay', color="factor_name",
                     orientation="v", hover_name="factor_name",
                     color_discrete_sequence=self.listt,
                     title="Direction of the Least important factors")

        self.fig.show()


        
importance=Importance()
importance.importance_factors()
importance.best_factors()
importance.best_factors_direction()
importance.worst_factors()
importance.worst_factors_direction()


df1=df1.reset_index()
df2=df2.reset_index()
plt.hist(X[df1['factor_name'][0]])
plt.title(df1['factor_name'][0]+" Distribution")
plt.show()

plt.hist(X[df1['factor_name'][1]])
plt.title(df1['factor_name'][1]+" Distribution")
plt.show()

plt.hist(X[df1['factor_name'][2]])
plt.title(df1['factor_name'][2]+" Distribution")
plt.show()

plt.hist(X[df2['factor_name'][0]])
plt.title(df2['factor_name'][0]+" Distribution")
plt.show()

plt.hist(X[df2['factor_name'][1]])
plt.title(df2['factor_name'][1]+" Distribution")
plt.show()

plt.hist(X[df2['factor_name'][2]])
plt.title(df2['factor_name'][2]+" Distribution")
plt.show()




_data = {df1['factor_name'][0]: data[df1['factor_name'][0]],
        df1['factor_name'][1]: data[df1['factor_name'][1]],
        df1['factor_name'][2]: data[df1['factor_name'][2]]}

X = pd.DataFrame(_data)

from sklearn import tree
from sklearn.tree import _tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Getting rules
from sklearn.tree import export_text
r = export_text(clf, feature_names=df1['factor_name'].tolist(), decimals=10)
#print(r)

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 10)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 10)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],10))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rules += [rule]
        
    n=len(rules)
    for i in range(n):
        print(rules[i])
        print(' ')
    
    return print('Completed')

print(get_rules(clf, df1['factor_name'].tolist(), [0,1,2,3]))
lines = r
with open('AllCases.txt', 'w') as f:
    for line in lines:
        f.write(line)
        
#Model score
print('The logistic model train success rate: ', clf.score(X, y)*100,'%')
print('')
print('The logistic model prediction outputs: ')

print(clf.predict(X))