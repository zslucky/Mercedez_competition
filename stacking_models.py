import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Stacking all model results
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

y_mean = np.mean(train['y'])

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

y_train = train["y"]
id_test = test["ID"]

'''
  Do stack
'''
random_forest_feature_df = pd.read_csv('./random_forest_feature_model.csv')
xgboost_feature_df = pd.read_csv('./xgboost_feature_model.csv')

random_forest_test_df = pd.read_csv('./random_forest_test_model.csv')
xgboost_test_df = pd.read_csv('./xgboost_test_model.csv')

xgb_params = {
    'eta': 0.005,
    'max_depth': 3,
    'subsample': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,
    'silent': 1
}

# random_forest_feature_df.drop('ID', axis=1, inplace=True)
# xgboost_feature_df.drop('ID', axis=1, inplace=True)

# random_forest_test_df.drop('ID', axis=1, inplace=True)
# xgboost_test_df.drop('ID', axis=1, inplace=True)

train['M1'] = random_forest_feature_df['y']
train['M2'] = xgboost_feature_df['y']

test['M1'] = random_forest_test_df['y']
test['M2'] = xgboost_test_df['y']

# x_train = pd.DataFrame({'M1': random_forest_feature_df['y'], 'M2': xgboost_feature_df['y']})
# x_test = pd.DataFrame({'M1': random_forest_test_df['y'], 'M2': xgboost_test_df['y']})

train.drop(['ID', 'y'], axis=1, inplace=True)
test.drop(['ID'], axis=1, inplace=True)

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test)

cv_result = xgb.cv(xgb_params,
                  dtrain,
                  num_boost_round=2500,
                  early_stopping_rounds=100,
                  verbose_eval=20,
                  show_stdv=False
                 )

num_boost_rounds = len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)


'''
  Show r2 score on train data
'''
y_pred_train = model.predict(dtrain)
y_train_score = r2_score(y_train, y_pred_train)
print('Train data r2 score = ', y_train_score)


y_test_pred = model.predict(dtest)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_test_pred
sub.to_csv('stacked_model_sub.csv', index=False)