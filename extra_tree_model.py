import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pandas as pd

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train[train['y'] > 200] = 200

'''
  Treat object values
'''
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

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
# estimator = ExtraTreesRegressor()

# param_grid = [
#   {'n_estimators': [5, 8, 10, 15, 20]},
#   {'max_depth': [5, 6, 7]}
# ]
# reg = GridSearchCV(estimator, param_grid, cv=cv, verbose=5)
# reg.fit(train, label_train)

# print('best_estimator_ = ', reg.best_estimator_)


et_params = {
  'n_estimators': 10,
  'max_depth': 5,
  'min_samples_leaf': 1,
  'min_samples_split': 2
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

  reg = ExtraTreesRegressor(**et_params)
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
average_importances = np.average(feature_importances, axis=0)
cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame({
  'features': cols,
   'ET_feature_importances': average_importances
  })

importances_df = feature_dataframe.sort_values('ET_feature_importances', ascending=False)
print(importances_df[:10])

'''
  Merge the test prediction data by average
'''
sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = np.average(y_preds, axis=0)

sub.to_csv('extra_tree_test_model.csv', index=False)

'''
'''
mode_random_forest = pd.DataFrame({'ID': id_train,'y': y_valid_preds})
mode_random_forest.to_csv('extra_tree_feature_model.csv', index=False)