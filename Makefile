.PHONE: requirements features

# Generate a requirements.txt file from pyproject.toml if you work with Poetry
requirements:
	python -m pip install --upgrade pip
	pip-compile -o requirements.txt pyproject.toml --resolver=backtracking

# generates a new batch of features and stores them in the feature store
features:
	poetry run python scripts/feature_pipeline.py

# trains a new model and stores it in the model registry
training:
	poetry run python scripts/training_pipeline.py

# generates predictions and stores them in the feature store
inference:
	poetry run python scripts/inference_pipeline.py

# starts the streamlit app
frontend:
	poetry run streamlit run src/frontend.py

monitoring:
	poetry run streamlit run src/frontend_monitoring.py