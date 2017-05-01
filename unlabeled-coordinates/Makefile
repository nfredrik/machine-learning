src_dir := src

python ?= python3
virtualenv_dir ?= pyenv
pip := $(virtualenv_dir)/bin/pip
linter := $(virtualenv_dir)/bin/flake8
coverage := $(virtualenv_dir)/bin/coverage
pytest := $(virtualenv_dir)/bin/py.test
py_requirements ?= requirements/prod.txt requirements/dev.txt


.PHONY: test
test: $(virtualenv_dir)
	PYTHONPATH=$(PYTHONPATH):. $(coverage) run \
		--source $(src_dir) --branch $(pytest) -s tests
	$(coverage) report -m

.PHONY: lint
lint: $(virtualenv_dir)
	$(linter) $(src_dir)

$(virtualenv_dir): $(py_requirements)
	$(python) -m venv $@
	for r in $^ ; do \
		$(pip) install -r $$r ; \
	done
