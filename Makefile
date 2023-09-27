# VARIABLES
PY = python3
DIR = src

# TASKS
install:
	@echo 'Install requirements...'
	pip install -r requirements.txt

transform:
	@echo 'Execute transform data...'
	$(PY) $(DIR)/transform_main.py

logreg:
	@echo 'Logistic Regression...'
	$(PY) $(DIR)/logreg_model.py

knn:
	@echo 'K Nearest Neighbors...'
	$(PY) $(DIR)/knn_model.py
