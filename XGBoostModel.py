import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, cross_validation
import scipy as sp
np.random.seed(1)
#sys.path.append('/som/calvinjs/Kaggle/xgboost-master/python/')
import xgboost as xgb

def logloss_mc(y_true, y_prob, epsilon=1e-15):
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


Submit = False

# import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')

# drop ids and get labels
train = train.drop('id', axis=1)
labels = train.target.values
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)

#scaler = preprocessing.StandardScaler()
#train = scaler.fit_transform(train)

if Submit == False:
	sss = cross_validation.StratifiedShuffleSplit(labels,n_iter=1,test_size=0.05,train_size=None,random_state=0)
	for train_index, test_index in sss:
		print("TRAIN:", train_index, "TEST:", test_index)
		rest, holdout = train.loc[train_index], train.loc[test_index]
		labels_rest, labels_holdout = labels[train_index], labels[test_index]
	'''
	msk = np.random.rand(len(train)) < 0.95
	rest = train[msk]
	holdout = train[~msk]
        labels_rest = labels[msk]
        labels_holdout = labels[~msk]
  	'''
	scaler = preprocessing.StandardScaler()
	rest = scaler.fit_transform(rest)
	holdout = scaler.transform(holdout)

else:
	rest = train
	labels_rest = labels
        scaler = preprocessing.StandardScaler()
        rest = scaler.fit_transform(rest)
        test = scaler.transform(test)
	

# encode labels
lbl_enc = preprocessing.LabelEncoder()
labels_rest = lbl_enc.fit_transform(labels_rest)
dtrain = xgb.DMatrix(rest, label=labels_rest)

if Submit == False:
	labels_holdout = lbl_enc.fit_transform(labels_holdout)
	dtest = xgb.DMatrix(holdout)
else:
	dtest = xgb.DMatrix(test)

#'''
param = {'eta':0.05,'min_child_weight':5.5,'max_delta_step':0.45,'max_depth':12,'silent':1, 'objective':'multi:softprob', 'nthread':60, 'eval_metric':'mlogloss','num_class':9,'subsample':1,'colsample_bytree':0.5,'gamma':0.5}
num_round = 820
bst = xgb.train(param, dtrain, num_round)

pred_test = bst.predict( dtest )

'''
# train a random forest classifier
#rf = ensemble.RandomForestClassifier(n_jobs=60, n_estimators=1000,oob_score=True)
rf = ensemble.RandomForestClassifier(n_estimators=1000,min_samples_split=1,min_samples_leaf=1,oob_score=True,n_jobs=60,max_features=50)

rf.fit(rest, labels_rest)
pred_test = rf.predict_proba(holdout)
#pred_test = clf.predict_proba(test)
'''
if Submit == False:
	error = logloss_mc(labels_holdout,pred_test)
	print error
else:
	pred_test = pd.DataFrame(pred_test, index=sample.id.values, columns=sample.columns[1:])
	pred_test.to_csv('18_XGB_scaler_smallchange.csv', index_label='id')
