import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
#try RandomForests & neural network if tree doesnt work
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)

data = pd.read_csv('train.csv')
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Embarked'], inplace = True, axis = 1)
data['Age'].fillna(data['Age'].mean(), inplace = True)
#print(data.Age.isnull().sum())

target = pd.DataFrame(data.Survived)
data.drop(['Survived'], axis = 1, inplace = True)
data = pd.get_dummies(data)

scaler = MinMaxScaler()
#print(type(scaler.fit(data)))
scaler.fit(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, stratify = target, random_state = 0)

test_acc = []
train_acc = []
C_settings = np.linspace(0.1, 10, 100)
max_test_acc = 0
optimal_C = 0

# Logistic Regression
for i in C_settings:
    logreg = LogisticRegression(C = i, random_state = 0, n_jobs=-1)
    logreg.fit(X_train, np.ravel(y_train))
    te_acc = logreg.score(X_test, y_test)
    tr_acc = logreg.score(X_train, y_train)
    test_acc.append(te_acc)
    train_acc.append(tr_acc)
    if max_test_acc < te_acc and te_acc < tr_acc:
        max_test_acc = te_acc
        optimal_C = i
print('LogisticRegression:')
print('max_test_acc = ', max_test_acc)
print('optimal_C = ', optimal_C)
plt.plot(C_settings, test_acc, marker = '.', label = 'test_acc')
plt.plot(C_settings, train_acc, marker = 'x', label = 'train_acc')
plt.legend()
plt.grid()
plt.show()

# KNeighbors Classifier
test_acc.clear()
train_acc.clear()
neighbor_settings = range(1, 13)
max_test_acc = 0
optimal_neighbors = 1
for i in neighbor_settings:
    knnc = KNeighborsClassifier(n_neighbors = i, n_jobs=-1)
    knnc.fit(X_train, np.ravel(y_train))
    te_acc = knnc.score(X_test, y_test)
    tr_acc = knnc.score(X_train, y_train)
    test_acc.append(te_acc)
    train_acc.append(tr_acc)
    if max_test_acc < te_acc and te_acc < tr_acc:
        max_test_acc = te_acc
        optimal_neighbors = i
print('\nKNeighbors Classification:')
print('max_test_acc = ', max_test_acc)
print('optimal_neighbors = ', optimal_neighbors)
plt.plot(neighbor_settings, test_acc, marker = '.', label = 'test_acc')
plt.plot(neighbor_settings, train_acc, marker = 'x', label = 'train_acc')
plt.legend()
plt.grid()
plt.show()

# Decision Tree Classifier
test_acc.clear()
train_acc.clear()
leaf_settings = range(2, 501)
max_test_acc = 0
optimal_leaves = 0
for i in leaf_settings:
    tree = DecisionTreeClassifier(max_leaf_nodes=i, random_state=0)
    tree.fit(X_train, np.ravel(y_train))
    te_acc = tree.score(X_test, y_test)
    tr_acc = tree.score(X_train, y_train)
    test_acc.append(te_acc)
    train_acc.append(tr_acc)
    if max_test_acc < te_acc and te_acc < tr_acc:
        max_test_acc = te_acc
        optimal_leaves = i
print('\nDecision Tree Classification:')
print('max_test_acc = ', max_test_acc)
print('optimal_leaves = ', optimal_leaves)
plt.plot(leaf_settings, test_acc, marker = '.', label = 'test_acc')
plt.plot(leaf_settings, train_acc, marker = 'x', label = 'train_acc')
plt.legend()
plt.grid()
plt.show()

# Random Forest Classifier
test_acc.clear()
train_acc.clear()
leaf_settings = range(2, 101)
max_test_acc = 0
optimal_leaves = 0
for i in leaf_settings:
    forest = RandomForestClassifier(max_leaf_nodes = i, random_state = 0, n_jobs=-1)
    forest.fit(X_train, np.ravel(y_train))
    te_acc = forest.score(X_test, y_test)
    tr_acc = forest.score(X_train, y_train)
    test_acc.append(te_acc)
    train_acc.append(tr_acc)
    if max_test_acc < te_acc and te_acc < tr_acc:
        max_test_acc = te_acc
        optimal_leaves = i
print('\nRandom Forest Classification:')
print('max_test_acc = ', max_test_acc)
print('optimal_leaves = ', optimal_leaves)
plt.plot(leaf_settings, test_acc, marker = '.', label = 'test_acc')
plt.plot(leaf_settings, train_acc, marker = 'x', label = 'train_acc')
plt.legend()
plt.grid()
plt.show()

# Gradient Boosting Classifier
'''
test_acc.clear()
train_acc.clear()
n_estimators_settings = range(100, 106)
max_depth_settings = range(3, 8)
learning_rate_settings = np.linspace(0.1, 2, 200)
max_test_acc = 0
max_train_acc = 0
optimal_n_estimators = 0
optimal_max_depth = 0
optimal_learning_rate = 0
print('\nGradient Boosting Classification:')
for i in n_estimators_settings:
    for j in max_depth_settings:
        for k in learning_rate_settings:
            gb_clf = GradientBoostingClassifier(n_estimators = i, max_depth = j, learning_rate = k , random_state = 0)
            gb_clf.fit(X_train, np.ravel(y_train))
            te_acc = gb_clf.score(X_test, y_test)
            tr_acc = gb_clf.score(X_train, y_train)
            if max_test_acc < te_acc and te_acc < tr_acc:
                max_test_acc = te_acc
                max_train_acc = tr_acc
                optimal_n_estimators = i
                optimal_max_depth = j
                optimal_learning_rate = k
                print('n_estimators = ', optimal_n_estimators)
                print('max_depth = ', optimal_max_depth)
                print('learning_rate = ', optimal_learning_rate)
                print('train_acc = ', gb_clf.score(X_train, y_train))
                print('test_acc = ', gb_clf.score(X_test, y_test), '\n')
print('n_estimators = ', optimal_n_estimators)
print('max_depth = ', optimal_max_depth)
print('learning_rate = ', optimal_learning_rate)
print('train_acc = ', max_test_acc)
print('test_acc = ', max_train_acc)
'''
gb_clf = GradientBoostingClassifier(n_estimators = 100, max_depth = 5, learning_rate = 0.1127 , random_state = 0)
gb_clf.fit(X_train, np.ravel(y_train))
print('\nGradient Boosting Classification:')
print('n_estimators = ', gb_clf.n_estimators)
print('max_depth = ', gb_clf.max_depth)
print('learning_rate = ', gb_clf.learning_rate)
print('train_acc = ', gb_clf.score(X_train, y_train))
print('test_acc = ', gb_clf.score(X_test, y_test))


# Support Vector Classifier
print('\nSupport Vector Classifier:\n')
data = pd.read_csv('train.csv')
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Embarked'], inplace = True, axis = 1)
data['Age'].fillna(data['Age'].mean(), inplace = True)
target = pd.DataFrame(data.Survived)
data.drop(['Survived'], axis = 1, inplace = True)
data = pd.get_dummies(data)

print('data head before scaling:\n', data.head())
print('data tail before scaling:\n', data.tail())
#data = pd.DataFrame(scale(data))
scaler = MinMaxScaler()
print(type(scaler.fit(data)))
print('data head after scaling:\n', data.head())
print('data tail before scaling:\n', data.tail())

X_train, X_test, y_train, y_test = train_test_split(data, target, stratify = target, random_state = 0)
svm_clf = SVC()
svm_clf.fit(X_train, np.ravel(y_train))
print('train_acc = ', svm_clf.score(X_train, y_train))
print('test_acc = ', svm_clf.score(X_test, y_test))

'''
LogisticRegression:
max_test_acc =  0.7982062780269058
optimal_C =  1.3000000000000003

KNeighbors Classification:
max_test_acc =  0.7309417040358744
optimal_neighbors =  4

Decision Tree Classification:
max_test_acc =  0.8295964125560538
optimal_leaves =  33

Random Forest Classification:
max_test_acc =  0.852017937219731
optimal_leaves =  42

Gradient Boosting Classification:
n_estimators =  100
max_depth =  5
learning_rate =  0.1127
train_acc =  0.9775449101796407
test_acc =  0.8475336322869955

Support Vector Classifier
train_acc =  0.905688622754491
test_acc =  0.726457399103139
'''