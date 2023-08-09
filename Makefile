# VARIABLES
PY = python3
DIR = src

# TASKS
install:
	@echo "Install requirements..."
	pip install -r requirements.txt

run_data:
	@echo "Execute data testing..."
	$(PY) $(DIR)/data_main.py
