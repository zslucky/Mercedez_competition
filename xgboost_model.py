import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
import xgboost as xgb
import pandas as pd

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

'''
  Data clean
'''
train.drop(train[train["y"] > 250].index, inplace=True)

train = train.reset_index()
train.drop(['index'], axis=1, inplace=True)

'''
  Treat object values
'''
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

#save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]


label_train = train['y']
id_train = train['ID'].values
train.drop(['ID', 'y'], axis=1, inplace=True)
y_mean = np.mean(label_train)
id_test = test['ID'].values
test.drop(['ID'], axis=1, inplace=True)

'''
  Set xgb params
'''
xgb_params = {
    'eta': 0.005,
    'max_depth': 5,
    'subsample': 0.98,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

dtest = xgb.DMatrix(test)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

y_preds = []
y_valid_preds = np.array([])
best_nums = []
y_valid_scores = []
for foldID, (train_index, valid_index) in enumerate(kf.split(train)):
    # Split data
    x_train, x_valid = train.iloc[train_index], train.iloc[valid_index]
    y_train, y_valid = label_train.iloc[train_index], label_train.iloc[valid_index]

    dtrain = xgb.DMatrix(x_train, y_train)
    dvalid = xgb.DMatrix(x_valid)

    cv_result = xgb.cv(xgb_params,
                  dtrain,
                  num_boost_round=1500, # increase to have better results (~700)
                  early_stopping_rounds=50,
                  verbose_eval=20,
                  show_stdv=False
                 )

    num_boost_rounds = len(cv_result)
    best_nums.append(num_boost_rounds)
    print('foldID: ', foldID, ' num_boost_rounds=' + str(num_boost_rounds))

    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

    '''
      Valid
    '''
    y_pred_valid = model.predict(dvalid)
    y_valid_score = r2_score(y_valid, y_pred_valid)
    print('foldID: ', foldID ,' r2 score = ', y_valid_score)
    y_valid_scores.append(y_valid_score)

    y_valid_preds = np.concatenate((y_valid_preds, y_pred_valid), axis=0)

    '''
      For test
    '''
    y_pred = model.predict(dtest)

    y_preds.append(y_pred)


# dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
# dtest = xgb.DMatrix(test)

# cv_result = xgb.cv(xgb_params,
#                   dtrain,
#                   num_boost_round=1000, # increase to have better results (~700)
#                   early_stopping_rounds=50,
#                   verbose_eval=10,
#                   show_stdv=False
#                  )

# num_boost_rounds = len(cv_result)
# print('num_boost_rounds=' + str(num_boost_rounds))

# # num_boost_rounds = 1365
# # train model
# model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)


# '''
#   Predict train data
# '''
# y_pred_train = model.predict(dtrain)
# print(r2_score(y_train, y_pred_train))


# '''
#   Make test predict
# '''
# y_pred = model.predict(dtest)

'''
  Generate the sub file
'''

print('----Best num rounds----')
print(best_nums)
print('----Valid scores----')
print(y_valid_scores)


# sub = pd.DataFrame()
# sub['ID'] = id_test
# sub['y'] = y_pred
# sub.to_csv('xgboost_model_sub.csv', index=False)

'''
  Merge the test prediction data by average
'''
sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = np.average(y_preds, axis=0)

sub.to_csv('xgboost_test_model.csv', index=False)

'''
'''
mode_xgb = pd.DataFrame({'ID': id_train, 'y': y_valid_preds})
mode_xgb.to_csv('xgboost_feature_model.csv', index=False)