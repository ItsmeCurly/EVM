# Eulerian Video Magnification



## Installation/Setup

1. `git clone https://github.com/ItsmeCurly/EVM.git && cd evm/`
2. `curl -sSLk https://install.python-poetry.org | python3 - &&`
Ensure that poetry is running: `poetry version`
If poetry is not found, add it to .bashrc or environment variables

### Setup Virtualenv
1. python -m venv env
2. `. env/bin/activate` OR `\env\Scripts\activate.bat` OR `& env/Scripts/Activate.ps1`
3. `poetry lock`
4. `poetry install --no-root`

### Setup pre-commit
`pre-commit` is a program for all projects, not just Python projects, but requires a Python installation to run. `pre-commit` runs when `git commit` is  called, and will run whatever commands that are located within the `.pre-commit-config.yaml`. For this project, `pre-commit` runs to enforce an uncompromising style on the code between developers.
1. pre-commit install
2. pre-commit run --all-files
If you get a message saying pre-commit is not there, then you're either not in a virtual environment with pre-commit or the poetry install command failed somehow.
