FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG UID=1000
ARG GID=1000
ARG USER=angel
ARG JUPYTER_PASSWORD="langchain2024"

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y curl

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN addgroup --gid $GID $USER
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID $USER

USER $USER
WORKDIR /home/$USER

USER root

RUN pip3 install torch
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN pip install -qU \
    scipy \
    transformers \
    accelerate \
    einops \
    langchain \
    langchain-community \
    langchain-ollama \
    xformers \
    bitsandbytes

RUN pip3 install jupyter
RUN pip3 install jupyterlab

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN pip3 install --upgrade langchain
RUN pip3 install langgraph
RUN pip3 install tavily-python
RUN pip3 install langchain_huggingface

USER $USER

RUN jupyter-lab --generate-config
RUN echo "c.ServerApp.password='$(python3 -c "from jupyter_server.auth import passwd; print(passwd('$JUPYTER_PASSWORD'))")'" >> ~/.jupyter/jupyter_lab_config.py

RUN echo "#!/bin/bash\n ollama serve &\n source /etc/bash.bashrc\n jupyter lab --ip 0.0.0.0 --no-browser" > .entrypoint.sh
RUN chmod +x .entrypoint.sh
CMD ["./.entrypoint.sh"]
