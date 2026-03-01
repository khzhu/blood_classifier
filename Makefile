#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = blood-classifier
PYTHON_VERSION = 3.12.2
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests
## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	az storage blob download-batch -s blood-classifier/data/ \
		-d data/
	

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	az storage blob upload-batch -d blood-classifier/data/ \
		-s data/
	



## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: dataset
dataset:
	$(PYTHON_INTERPRETER) -m classifier.dataset

## Select features
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m classifier.features --target_name "$(target_name)"

## Train model
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m classifier.modeling.train --target_name "$(target_name)"

## Make predict
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m classifier.modeling.predict \
						  --model "$(model)" --target_name "$(target_name)"

## Make plots
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) -m classifier.plots \
						  --model "$(model)" --target_name "$(target_name)"

# Pipeline combines predict and plots
pipeline: train predict plots
	@echo "Pipeline finished!"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
