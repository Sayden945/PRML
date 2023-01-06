from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib as plt

bCancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(bCancer.data, bCancer.target, train_size=.8, random_state=0)

model = SVC()

param_grid = [{'C': [0.5, 0.1, 1, 5, 10],'kernel': ['linear'], 'class_weight':['balanced']},
              {'C': [0.5, 0.1, 1, 5, 10],'kernel': ['sigmoid'], 'class_weight':['balanced']},
              {'C': [0.5, 0.1, 1, 5, 10],'kernel': ['poly'], 'class_weight':['balanced']},
              {'C': [0.5, 0.1, 1, 5, 10],'kernel': ['rbf'], 'class_weight': ['balanced']}]

grs = GridSearchCV(model, param_grid)
grs.fit(X_train, y_train)
print("Best Hyper Parameters:",grs.best_params_)

y_pred = grs.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))
print("F1-score:",metrics.f1_score(y_test, y_pred, average = 'weighted'))



