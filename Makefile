# VARIABLES
PY = python3
DIR = src

# TASKS
install:
	@echo 'Install requirements...'
	pip install -r requirements.txt

trn_data:
	@echo 'Execute transform data...'
	$(PY) $(DIR)/transform_main.py

log_reg:
	@echo 'Logistic Regression...'
	$(PY) $(DIR)/logreg_model.py

k_nn:
	@echo 'K Nearest Neighbors...'
	$(PY) $(DIR)/knn_model.py

dec_tree:
	@echo 'Decision Tree Classifier...'
	$(PY) $(DIR)/dectree_model.py

ran_for:
	@echo 'Random Forest Classifier...'
	$(PY) $(DIR)/ranfor_model.py

xgboost:
	@echo 'XGBoost Classifier...'
	$(PY) $(DIR)/xgboost_model.py
