# Vision: from 0 to Anthropic

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
pyenv install 3.13
pyenv local 3.13
```

To run the notebooks locally:                                           
```sh                                                                     
poetry shell                                                            
jupyter lab                                                             
```

---
## Dependencies

1. OpenCV: https://pypi.org/project/opencv-python/
1. Jupyter: https://jupyter.org/install
1. Matplotlib: https://matplotlib.org/stable/install/index.html
1. tqdm: https://github.com/tqdm/tqdm
1. ipywidgets: https://ipywidgets.readthedocs.io/en/stable/user_install.html
1. ipympl: https://matplotlib.org/ipympl/

```sh
poetry add opencv-contrib-python
poetry add jupyterlab
poetry add matplotlib
poetry add tqdm
poetry add ipywidgets
poetry add ipympl
```


