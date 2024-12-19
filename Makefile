kernel:
	python3 -m pip install ipykernel
	python3 -m ipykernel install --user

setup-formatter:
	poetry add jupyterlab-code-formatter
	poetry add black isort
	# Go to Settings > Settings Editor > JSON Settings Editor (Top Right) > JupyterLab Code Formatter > {"formatOnSave": true}

jupyter:
	poetry run jupyter lab

convert_all:
	# jupytext doesn't preserve image.
	#@find . -name "*.ipynb" ! -path '*/.*' -exec poetry run jupytext --to md {} \;
	@find . -name "*.ipynb" ! -path '*/.*' -exec poetry run jupyter nbconvert --to markdown --output-dir=docs {} \;

# Similar to convert, but only convert the diff files.
convert:
	@poetry run jupyter nbconvert --to markdown --output-dir=docs $(shell git diff HEAD --name-only | grep .ipynb)


cover:
	@# The s prints the print output to stdout.
	poetry run coverage run -m pytest -s
	poetry run coverage report -m
	poetry run coverage html
	open htmlcov/index.html
