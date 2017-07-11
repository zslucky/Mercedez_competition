# Mercedez competition

My 2nd competition on kaggle

201st of 3927,  bronze medal

---

## Files

### Jupyter note book

* `Object columns Analysis Book.ipynb`: data analysis for object columns `X0, X1, X2, X3, X4, X5, X6, X8`.

### Main model files

* `adaboost_regressor_model.py`: stacked feature model by using adaboost regressor.
* `extra_tree_model.py`: stacked feature model by using adaboost regressor.
* `random_forest_regressor_model.py`: stacked feature model by using adaboost regressor.
* `gradient_boost_model.py`: stacked feature model by using adaboost regressor.
* `stacking_models.py`: using XGBoost to stack all model features, and combine with the other submission.

### Tool files

* `all_questions.json`: this is the probed test data from others guys, just used for test.
* `test_y_value.csv`: this is a csv file transformed from `all_question.json`.
* `test_data_update.py`: this script used to transform the json file to csv file.
* `test_data_compared.py`: this script used to compared every result csv to probed value, also show the r2 score.

### Online model

* `origin-script`: This script comes from [Hakeem's script](https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697)