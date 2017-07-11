import pandas as pd
from sklearn.metrics import r2_score

test_y_value = pd.read_csv('./test_y_value.csv')
rf_test_pred = pd.read_csv('./random_forest_test_model.csv')
ab_test_pred = pd.read_csv('./adaboost_test_model.csv')
gb_test_pred = pd.read_csv('./gradient_boost_test_model.csv')
et_test_pred = pd.read_csv('./extra_tree_test_model.csv')

stacked_test_pred = pd.read_csv('./stacked_model_sub.csv')
stacked_pipeline_test_pred = pd.read_csv('./stacked_ensemble_sub.csv')

origin_sc_test_pred = pd.read_csv('./online_model/origin_script_sub.csv')
stacked_pipeline_sc_test_pred = pd.read_csv('./stacked_ensemble_combine_sc_sub.csv')


merged_compared_values = test_y_value.join(stacked_test_pred.set_index('ID'), on='ID', how='left', rsuffix='_stack')
merged_compared_values = merged_compared_values.join(stacked_pipeline_test_pred.set_index('ID'), on='ID', how='left', rsuffix='_stack_pipeline')
merged_compared_values = merged_compared_values.join(origin_sc_test_pred.set_index('ID'), on='ID', how='left', rsuffix='_origin')
merged_compared_values = merged_compared_values.join(stacked_pipeline_sc_test_pred.set_index('ID'), on='ID', how='left', rsuffix='_spsc')
merged_compared_values = merged_compared_values.join(rf_test_pred.set_index('ID'), on='ID', how='left', rsuffix='_rf')
merged_compared_values = merged_compared_values.join(ab_test_pred.set_index('ID'), on='ID', how='left', rsuffix='_ab')
merged_compared_values = merged_compared_values.join(gb_test_pred.set_index('ID'), on='ID', how='left', rsuffix='_gb')
merged_compared_values = merged_compared_values.join(et_test_pred.set_index('ID'), on='ID', how='left', rsuffix='_et')


print(merged_compared_values)


y_rf_r2_score = r2_score(test_y_value['y'].values, merged_compared_values['y_rf'].values)

y_ab_r2_score = r2_score(test_y_value['y'].values, merged_compared_values['y_ab'].values)

y_gb_r2_score = r2_score(test_y_value['y'].values, merged_compared_values['y_gb'].values)

y_et_r2_score = r2_score(test_y_value['y'].values, merged_compared_values['y_et'].values)

y_origin_r2_score = r2_score(test_y_value['y'].values, merged_compared_values['y_origin'].values)

y_stack_pipeline_r2_score = r2_score(test_y_value['y'].values, merged_compared_values['y_stack_pipeline'].values)

y_stack_r2_score = r2_score(test_y_value['y'].values, merged_compared_values['y_stack'].values)

y_spsc_r2_score = r2_score(test_y_value['y'].values, merged_compared_values['y_spsc'].values)

print('rf scroe = ', y_rf_r2_score)
print('ab scroe = ', y_ab_r2_score)
print('gb scroe = ', y_gb_r2_score)
print('et scroe = ', y_et_r2_score)
print('origin scroe = ', y_origin_r2_score)
print('stack pipeline scroe = ', y_stack_pipeline_r2_score)
print('stack scroe = ', y_stack_r2_score)
print('spsc scroe = ', y_spsc_r2_score)