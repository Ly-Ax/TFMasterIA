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
