import pandas as pd

test_y_df = pd.read_json('./all_questions.json')

test_y_df_value = test_y_df[['id', 'answers']].values

test_value_list = {
  'ID': [],
  'y': []
}
for test_value in test_y_df_value:
  test_id = test_value[0]
  obj_arr = test_value[1]

  if(len(obj_arr) and obj_arr[0]['inside_public_lb'] == True):
    test_value_list['ID'].append(test_id)
    test_value_list['y'].append(obj_arr[0]['y_value'])


exits_prediction = pd.DataFrame(test_value_list)
exits_prediction.to_csv('test_y_value.csv', index=False)