# Use official Python base image (multi-arch: amd64, arm64)
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        avahi-daemon \
        dbus \
        git \
        build-essential \
        tmux \
        tmuxinator \
        curl \
        openssh-client \
        iputils-ping \
        libnss-mdns \
        scons \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install torch-geometric
RUN pip3 install pyyaml matplotlib networkx tqdm codetiming
RUN pip3 install rpi_ws281x

# Copy tmux configuration
COPY ./.tmux.conf /root/.tmux.conf

# Add killp function to .bashrc
RUN echo 'killp() {\n\
    if [ $# -eq 0 ]; then\n\
        echo "The command killp() needs an argument, but none was provided!"\n\
        return\n\
    else\n\
        pes=$1\n\
    fi\n\
    for child in $(ps -o pid,ppid -ax | \\\n\
        awk "{ if ( \\$2 == $pes ) { print \\$1 }}")\n\
        do\n\
            killp $child\n\
        done\n\
    kill -9 "$1" > /dev/null 2> /dev/null\n\
}\n' >> /root/.bashrc


# Set up RGB shield
RUN mkdir -p /root/workspace
WORKDIR /root/workspace
RUN git clone https://github.com/jgarff/rpi_ws281x.git
WORKDIR /root/workspace/rpi_ws281x

RUN sed -i 's/^#define WIDTH.*/#define WIDTH                   8/' main.c && \
    sed -i 's/^#define HEIGHT.*/#define HEIGHT                  4/' main.c
RUN scons

# Clone git repositories
WORKDIR /root/workspace
RUN git clone https://github.com/mkrizmancic/my_graphs_dataset.git
RUN git clone https://github.com/monikab11/distributed_computation.git

# Build and install
RUN cd /root/workspace/my_graphs_dataset && pip3 install -e .

WORKDIR /root/workspace/distributed_computation/simple_gnn

# Default command
CMD ["sh", "-c", "service dbus start && /usr/sbin/avahi-daemon --daemonize && exec /bin/bash"]
