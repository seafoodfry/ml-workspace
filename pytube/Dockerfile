FROM python:3.12

WORKDIR /tmp

RUN apt-get update -y && \
    apt-get install vim less wget gnupg build-essential -y

RUN apt-get update -y && \
    apt-get install bash-completion -y && \ 
    python3 -m venv /opt/venv

# Manually creating a user. The other option is to run with
# -u $(id -u ${USER}):$(id -g ${USER})
# Another good example: https://github.com/jupyter/docker-stacks/blob/b2efa3cd8ce7a2b900f76ff4a17e11a000666cea/images/docker-stacks-foundation/Dockerfile#L72
RUN useradd --no-log-init --create-home --shell /bin/bash --uid 1000 --no-user-group  user && \
    chown -R user /home/user && \
    chown -R user /opt/venv

USER user

WORKDIR /home/user

ENV PATH=/opt/venv/bin:$PATH
ENV PATH=$PATH:/home/user/.local/bin

COPY deps/requirements.txt .
RUN . /etc/profile.d/bash_completion.sh && \
    pip install -Ur requirements.txt

COPY main.py . 

CMD ["python", "-u", "main.py"]
