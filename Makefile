.PHONE: requirements features

# Generate a requirements.txt file from pyproject.toml if you work with Poetry
requirements:
	python -m pip install --upgrade pip
	pip-compile -o requirements.txt pyproject.toml --resolver=backtracking

# run the feature pipeline
features:
	poetry run python scripts/feature_pipeline.py