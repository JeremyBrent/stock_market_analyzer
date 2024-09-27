venv=venv
software_path=.

configure:
	@# Create Python venv
	python3.9 -m venv $(software_path)/$(venv)

	@# Install requirements
	$(software_path)/$(venv)/bin/pip3 install -r requirements.txt



test:
	source $(software_path)/$(venv)/bin/activate; python3 --version; which python3

install: configure test