# LLMs


## Setup

We installed poetry as
```
VENV_PATH=~/.poetry
python3 -m venv $VENV_PATH
$VENV_PATH/bin/pip install poetry
```

In the current working directory we run:
```
export PATH=~/.poetry/bin/:$PATH
```

We had to manage the python version in order to get PyTorch installed.
We used `pyenv` for this:

```
pyenv versions
* system (set by /Users/user/.pyenv/version)
  3.10.14
```

```
pyenv install 3.12
pyenv local 3.12
```

When we saw that we still kept getting
```
Unable to find installation candidates for torch (2.5.1)
```

Errors, we switched to a plain old venv:
```
python3 -m venv .venv
source .venv/bin/activate
```


To run the notebooks locally:
```
poetry shell
jupyter lab
```

## Dependencies

Explicitly documenting dependencies because we had to start from scratch a handful of times:

```
pip install jupyterlab                     # https://jupyter.org/install
pip install Pympler                        # https://pypi.org/project/Pympler/
pip install tiktoken                       # https://github.com/openai/tiktoken
pip3 install torch torchvision torchaudio  # https://pytorch.org/get-started/locally/
```
