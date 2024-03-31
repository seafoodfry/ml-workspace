FROM python:3.12

WORKDIR /home/user

RUN apt-get update -y && \
    apt-get install bash-completion vim less ffmpeg -y && \
    . /etc/profile.d/bash_completion.sh

# Manually creating a user. The other option is to run with
# -u $(id -u ${USER}):$(id -g ${USER})
# Another good example: https://github.com/jupyter/docker-stacks/blob/b2efa3cd8ce7a2b900f76ff4a17e11a000666cea/images/docker-stacks-foundation/Dockerfile#L72
RUN useradd --no-log-init --create-home --shell /bin/bash --uid 1000 --no-user-group  user && \
    chown -R user /home/user
USER user
ENV PATH=$PATH:/home/user/.local/bin
RUN pip install -U openai-whisper
