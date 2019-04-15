from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("data.csv")
x = df.values
np.random.shuffle(x)

y = x[:, 4]
x = x[:, :4]

scaler = StandardScaler().fit(x)
x = scaler.transform(x)
print(scaler.mean_ ,scaler.scale_)

# with open('save/scaler.pickle','wb') as output:
#     s = pickle.dump(scaler, output)

x_train = x[:14500]
x_test = x[14500:]
y_train = y[:14500]
y_test = y[14500:]

weights = y * 79
weights += 1


# clf = SVC(kernel='linear', verbose = 1, max_iter=100000)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=11, random_state=0, min_samples_leaf=60, min_samples_split=1800)
clf.fit(x_train, y_train, sample_weight=weights)

# param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}
# gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=11,
#                                      max_features='sqrt', subsample=0.8, random_state=10), 
#                        param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(x,y)
# print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)

# fin = open('save/clf.pickle', 'rb')
# clf = pickle.load(fin)

# print(clf.predict(scaler.transform([[44,75.38359568,159,62]])))
# print(clf.predict(scaler.transform([[12,187,500,153]])))
# print(clf.predict(scaler.transform([[220,19,100,100]])))
# print(clf.predict(scaler.transform([[200,200,200,100]])))
# print(clf.predict(scaler.transform([[200,200,500,385]])))
# print(clf.predict(scaler.transform([[1,200,500,900]])))
# print(clf.predict(scaler.transform([[1,75,500,900]])))
# print(clf.predict(scaler.transform([[72,1.4,500,900]])))
# print(clf.predict(scaler.transform([[72,1.4,50,90]])))
# print(clf.predict(scaler.transform([[29,152,156,156]])))

y1 = clf.predict(x_test)
print(precision_recall_fscore_support(y_test, y1, average='binary'))
print(roc_auc_score(y_test, y1))


with open('save/gbdt.pickle','wb') as output:
    s = pickle.dump(clf, output)

