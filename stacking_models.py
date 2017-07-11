import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.metrics import r2_score

RANDOM_STATE = 20170710
n_comp = 12

# Stacking all model results
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# test.loc[[3913, 3718, 3281, 2914], 'X0'] = 'q'
# test.loc[[311,153], 'X0'] = 'aa'

# test.loc[[4187, 4177, 4161], 'X2'] = 'ar'
# test.loc[[3911], 'X2'] = 'o'
# test.loc[[3323], 'X2'] = 'c'
# test.loc[[3073, 2930], 'X2'] = 'aa'
# test.loc[[2473], 'X2'] = 'l'

# test.loc[test['X5'].isin(['a', 'b', 't', 'z']), 'X5'] = 'u'

y_mean = np.mean(train['y'])

'''
  Data clean
'''
# train[train['y'] > 200] = 200

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

train.drop(['y'], axis=1, inplace=True)


# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
pca2_results_train = pca.fit_transform(train)
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=RANDOM_STATE)
ica2_results_train = ica.fit_transform(train)
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=RANDOM_STATE)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=RANDOM_STATE)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

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

'''
  Do stack
'''
random_forest_feature_df = pd.read_csv('./random_forest_feature_model.csv')
extra_tree_feature_df = pd.read_csv('./extra_tree_feature_model.csv')
adaboost_feature_df = pd.read_csv('./adaboost_feature_model.csv')
gradient_boost_feature_df = pd.read_csv('./gradient_boost_feature_model.csv')
# svr_feature_df = pd.read_csv('./svr_feature_model.csv')

random_forest_test_df = pd.read_csv('./random_forest_test_model.csv')
extra_tree_test_df = pd.read_csv('./extra_tree_test_model.csv')
adaboost_test_df = pd.read_csv('./adaboost_test_model.csv')
gradient_boost_test_df = pd.read_csv('./gradient_boost_test_model.csv')
# svr_test_df = pd.read_csv('./svr_test_model.csv')


# random_forest_feature_df.drop('ID', axis=1, inplace=True)
# xgboost_feature_df.drop('ID', axis=1, inplace=True)

# random_forest_test_df.drop('ID', axis=1, inplace=True)
# xgboost_test_df.drop('ID', axis=1, inplace=True)

# train['M1'] = random_forest_feature_df['y']
train['M2'] = extra_tree_feature_df['y']
# train['M3'] = adaboost_feature_df['y']
train['M4'] = gradient_boost_feature_df['y']
# train['M5'] = svr_feature_df['y']

# test['M1'] = random_forest_test_df['y']
test['M2'] = extra_tree_feature_df['y']
# test['M3'] = adaboost_feature_df['y']
test['M4'] = gradient_boost_test_df['y']
# test['M5'] = svr_test_df['y']

# train = train.loc[:, ['M1', 'M2', 'M3', 'M4']]
# test = test.loc[:, ['M1', 'M2', 'M3', 'M4']]

xgb_params = {
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.98,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}



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

# num_boost_rounds = 702

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)


'''
  Show r2 score on train data
'''
y_pred_train = model.predict(dtrain)
y_train_score = r2_score(y_train, y_pred_train)
print('Train data on XGBoost r2 score = ', y_train_score)


y_test_pred = model.predict(dtest)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_test_pred
sub.to_csv('stacked_model_sub.csv', index=False)


'''
  Add stacking pipeline
'''
print('-----pipeline ensemble-----')
finaltrainset = train.values
finaltestset = test.values

class StackingEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        print('fit shape = ', X.shape)
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()
)

stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)
train_results = stacked_pipeline.predict(finaltrainset)

'''
  combine the prediction
'''
stacked_pipeline_ratio = 0.1
model_ratio = 0.9

new_train_pred = y_pred_train * model_ratio + train_results * stacked_pipeline_ratio
new_test_pred = y_test_pred * model_ratio + results * stacked_pipeline_ratio

ensemble_train_score = r2_score(y_train, new_train_pred)
print('Train data on ensemble r2 score = ', ensemble_train_score)

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = new_test_pred
sub.to_csv('stacked_ensemble_sub.csv', index=False)

'''
combine the origin best num
'''
origin_sc_test_pred = pd.read_csv('./online_model/origin_script_sub.csv')

sc_ratio = 0.25
model_ratio = 0.75

combined_test_pred = new_test_pred * model_ratio + origin_sc_test_pred['y'] * sc_ratio

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = combined_test_pred
sub.to_csv('stacked_ensemble_combine_sc_sub.csv', index=False)