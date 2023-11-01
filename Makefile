# VARIABLES
PY = python
DIR = src

# TASKS
test:
	@echo 'Hello, World!!!'

inst:
	@echo 'Install requirements...'
	pip install -r requirements.txt

pre:
	@echo 'Execute transform data...'
	$(PY) $(DIR)/transform_main.py

lr:
	@echo 'Logistic Regression...'
	$(PY) $(DIR)/logreg_model.py

knn:
	@echo 'K Nearest Neighbors...'
	$(PY) $(DIR)/knn_model.py

dtc:
	@echo 'Decision Tree Classifier...'
	$(PY) $(DIR)/dectree_model.py

rfc:
	@echo 'Random Forest Classifier...'
	$(PY) $(DIR)/ranfor_model.py

xgb:
	@echo 'XGBoost Classifier...'
	$(PY) $(DIR)/xgboost_model.py
