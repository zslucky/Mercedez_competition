import pandas as pd

# Stacking all model results
model_1 = pd.read_csv('./xgboost_model_sub.csv')
model_2 = pd.read_csv('./random_forest_model_sub.csv')

model_1['y'] = model_1['y'] * 0.8 + model_2['y'] * 0.2
model_1.to_csv('sub-stacked-models-results.csv', index=False)