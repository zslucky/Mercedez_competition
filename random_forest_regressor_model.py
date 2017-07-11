import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pandas as pd

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train[train['y'] > 200] = 200

'''
  Data clean
'''
# train.drop(train[train["y"] > 250].index, inplace=True)

# train = train.reset_index()
# train.drop(['index'], axis=1, inplace=True)

'''
  Treat object values
'''
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# n_comp = 12

# # tSVD
# tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
# tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
# tsvd_results_test = tsvd.transform(test)

# # PCA
# pca = PCA(n_components=n_comp, random_state=420)
# pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
# pca2_results_test = pca.transform(test)

# # ICA
# ica = FastICA(n_components=n_comp, random_state=420)
# ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
# ica2_results_test = ica.transform(test)

# # GRP
# grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
# grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
# grp_results_test = grp.transform(test)

# # SRP
# srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
# srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
# srp_results_test = srp.transform(test)

# #save columns list before adding the decomposition components

# usable_columns = list(set(train.columns) - set(['y']))

# # Append decomposition components to datasets
# for i in range(1, n_comp + 1):
#     train['pca_' + str(i)] = pca2_results_train[:, i - 1]
#     test['pca_' + str(i)] = pca2_results_test[:, i - 1]

#     train['ica_' + str(i)] = ica2_results_train[:, i - 1]
#     test['ica_' + str(i)] = ica2_results_test[:, i - 1]

#     train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
#     test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

#     train['grp_' + str(i)] = grp_results_train[:, i - 1]
#     test['grp_' + str(i)] = grp_results_test[:, i - 1]

#     train['srp_' + str(i)] = srp_results_train[:, i - 1]
#     test['srp_' + str(i)] = srp_results_test[:, i - 1]



label_train = train['y']
id_train = train['ID'].values
train.drop(['ID', 'y'], axis=1, inplace=True)

# y_mean = np.mean(y_train)
id_test = test['ID'].values
test.drop(['ID'], axis=1, inplace=True)

'''
  Using GridSearchCV to find best params
'''
# cv = KFold(n_splits=10, shuffle=True, random_state=20170704)
# estimator = RandomForestRegressor()

# param_grid = [
#   {'n_estimators': [250, 500]},
#   {'max_depth': [3, 4, 5]}
# ]
# reg = GridSearchCV(estimator, param_grid, cv=cv, verbose=5)
# reg.fit(train, label_train)

# print('best_estimator_ = ', reg.best_estimator_)

rf_params = {
  'n_estimators': 10,
  'max_depth': 4,
  'min_samples_split': 2,
  'min_samples_leaf': 1
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

y_preds = []
feature_importances = []
y_valid_preds = np.array([])

for foldID, (train_index, valid_index) in enumerate(kf.split(train)):
  # Split data
  x_train, x_valid = train.iloc[train_index], train.iloc[valid_index]
  y_train, y_valid = label_train.iloc[train_index], label_train.iloc[valid_index]

  # x_valid.to_csv(str(foldID) + '_x_rf.csv')
  # y_valid.to_csv(str(foldID) + '_y_rf.csv')

  reg = RandomForestRegressor(**rf_params)
  reg.fit(x_train, y_train)

  feature_importances.append(reg.feature_importances_)

  '''
    Show valid r2 score
  '''
  y_pred_valid = reg.predict(x_valid)
  valid_r2_score = r2_score(y_valid, y_pred_valid)
  print('fold ', foldID , ' valid r2 score = ', valid_r2_score)

  '''
    Show raw train r2 score
  '''
  y_pred_train = reg.predict(train)
  train_r2_score = r2_score(label_train, y_pred_train)
  print('fold ', foldID , ' raw train r2 score = ', train_r2_score)

  '''
    Predict test data and collect predict data
  '''
  y_preds.append(reg.predict(test))
  y_valid_preds = np.concatenate((y_valid_preds, y_pred_valid), axis=0)

'''
  Print the feature importances
'''
rf_average_importances = np.average(feature_importances, axis=0)
cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame({
  'features': cols,
   'RF_feature_importances': rf_average_importances
  })

importances_df = feature_dataframe.sort_values('RF_feature_importances', ascending=False)
print(importances_df[:10])

'''
  Merge the test prediction data by average
'''
sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = np.average(y_preds, axis=0)

sub.to_csv('random_forest_test_model.csv', index=False)

'''
'''
mode_random_forest = pd.DataFrame({'ID': id_train,'y': y_valid_preds})
mode_random_forest.to_csv('random_forest_feature_model.csv', index=False)

