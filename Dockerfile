FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV NODE_MAJOR=20

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3.9 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    google-perftools \
    ca-certificates curl gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN mkdir -p /etc/apt/keyrings 
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg

RUN echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list > /dev/null
RUN apt-get update && apt-get install nodejs -y

COPY ./server/requirements.txt /code/requirements.txt

# Download and install UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN chmod +x /uv-installer.sh && \
    /uv-installer.sh && \
    rm /uv-installer.sh

ENV PATH="/root/.local/bin:$PATH"

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Install dependencies using UV as root
RUN uv pip install --no-cache --system --index-strategy=unsafe-best-match -r /code/requirements.txt 

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:/root/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1 \
    SYSTEM=spaces

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
CMD ["./build-run.sh"]