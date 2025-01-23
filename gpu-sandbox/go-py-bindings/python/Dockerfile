FROM python:3.13

# Same values as docker-stack-foundation.
# See
# https://github.com/jupyter/docker-stacks/blob/aabaa58072623dcc62a71c9e84e3501ab11f5e9d/images/docker-stacks-foundation/Dockerfile#L11-L13
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# Install system dependencies and tools.
USER root

# Create NB_USER with name jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN useradd --no-log-init --create-home --shell /bin/bash --uid "${NB_UID}" --no-user-group "${NB_USER}" && \
    chown -R ${NB_USER} "/home/${NB_USER}"

# libcurl: https://ec.haxx.se/install/linux.html
RUN apt-get update -y && \
    apt-get install -y libcurl4-openssl-dev clang-format


##############################
### Non-Root Configuration ###
##############################
USER $NB_UID

WORKDIR /home/${NB_USER}/work

#######################
# Poetry Installation #
#######################
ENV POETRY_VERSION=1.8.4
ENV POETRY_HOME=/home/${NB_USER}/.poetry
RUN python3 -m venv ${POETRY_HOME} && \
    $POETRY_HOME/bin/pip install poetry==${POETRY_VERSION}

# Add Poetry to PATH.
ENV PATH="${POETRY_HOME}/bin:${PATH}"

#####################
# App Installation #
#####################
# Set Poetry environment variables to create a virtual environment.
# Not creating a virtualenv will lead poetry to want to install things in a global scope.
# So we just deal with this by sourcing the virtualenv manually right before we give control
# back to the user.
# See https://github.com/python-poetry/poetry/issues/1214
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_CACHE_DIR="/home/${NB_USER}/.cache/pypoetry"
COPY pyproject.toml poetry.lock /home/jovyan/work/
RUN poetry install --no-root --no-interaction --no-ansi
