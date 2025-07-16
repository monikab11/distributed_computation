# Distributed Graph Neural Network (GNN) Computation

A distributed implementation of Graph Neural Networks that enables computation across multiple nodes in a network, with support for Raspberry Pi devices with LED matrix visualization.

## Overview

This project implements a distributed GraphSAGE neural network where:
- **Central coordinator** distributes model configuration and test data to worker nodes
- **Worker nodes** perform local GNN computations and communicate with neighboring nodes
- **LED visualization** provides real-time feedback on Raspberry Pi devices
- **Docker support** for easy deployment across different platforms

## Architecture
Each node:
1. Receives initial configuration from central coordinator
2. Performs local GraphSAGE computations
3. Exchanges feature vectors with neighboring nodes
4. Visualizes progress on LED matrix (Raspberry Pi only)

## Features

- **Distributed GNN Processing**: Split GraphSAGE computation across multiple nodes
- **Network Communication**: Asynchronous TCP communication between nodes
- **LED Visualization**: Real-time computation progress display on Raspberry Pi LED matrices
- **Flexible Configuration**: YAML-based configuration for different deployment scenarios
- **Docker Support**: Containerized deployment for consistent environments
- **Cross-Platform**: Runs on both x86 and ARM architectures

## Project Structure

```
├── central.py          # Central coordinator for distributing configs
├── node.py             # Distributed node implementation
├── gnn_splitter.py     # Handles modifying existing models for distributed computation
├
├── train_gnn.py        # Prepares a simple model for demonstration, not needed if model is already available
├── dataset.py          # Dataset used by training script, not needed if dataset is already available
├── led_matrix.py       # LED matrix visualization (Raspberry Pi)
├── test_distributed.py # Testing the distributed computation concept
├
├── config/              # Configuration files
│   ├── local.yaml       # Local testing configuration
│   ├── remote.yaml      # Remote Raspberry Pi configuration
│   ├── mixed.yaml       # Mixed local/remote configuration
│   ├── model.pth        # Trained model weights - not saved on GitHub
│   └── test_dataset.pth # Test dataset - not saved on GitHub
├── Dataset/            # Processed graph datasets - not saved on GitHub
├── docker/             # Docker configuration
│   ├── Dockerfile
│   ├── run_docker.sh
│   └── start_docker.sh
```

## Installation

### Docker Installation (recommended)

```bash
cd docker
docker build -t rpi_dist_gnn .
./run_docker.sh
```

### Manual Installation
#### Prerequisites

- Python 3.10+
- PyTorch
- PyTorch Geometric
- NetworkX
- PyYAML

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pyyaml matplotlib networkx tqdm codetiming
```

#### Raspberry Pi Additional Dependencies

```bash
pip install rpi_ws281x  # For LED matrix control
```



## Usage

### 1. Training a Model

Save a model from your training pipeline, or use the provided training script to generate a simple GraphSAGE model.

```bash
python train_gnn.py
```

This creates:
- `config/model.pth` - Trained model weights
- `config/test_dataset.pth` - Test dataset for distributed inference

### 2. Configuration

Choose or modify a configuration file in `config/`:

**Local Testing (`local.yaml`)**:
```yaml
nodes:
  node0: [127.0.0.1, 5000]
  node1: [127.0.0.1, 5001]
  # ...
```

**Remote Raspberry Pi (`remote.yaml`)**:
```yaml
nodes:
  node0: [rpi0.local, 5000]
  node1: [rpi1.local, 5001]
  # ...
```

### 3. Running Distributed Computation

#### a. Automated Execution (recommended)
All nodes can be automatically started using `tmuxinator`:
```bash
tmuxinator start gnn <local|remote|mixed>
```
See the instructions below for navigating within the tmux session.

#### b. Manual Execution

**Start worker nodes** (on each device):
```bash
python node.py <node_id>
```

Example for node 0:
```bash
python node.py 0
```

**Start central coordinator**:
```bash
python central.py <config_name>
```

Example using local configuration:
```bash
python central.py local.yaml
```



## Docker Deployment

Start the Docker container on your computer. After that, all usage steps are the same as manual execution.
```bash
cd docker
./start_docker.sh
```



## Configuration Files

### Node Configuration
- `nodes`: Dictionary mapping node IDs to [host, port] pairs
- `model.path`: Path to trained model file
- `test_data.path`: Path to test dataset
- `test_data.index`: Index of test graph to use

### Available Configurations
- `local.yaml`: All nodes on localhost (testing)
- `remote.yaml`: Raspberry Pi cluster deployment
- `mixed.yaml`: Mixed local and remote nodes

## LED Matrix Visualization

On Raspberry Pi devices with LED matrices, the system provides real-time visualization:

- **Progress indication**: Shows computation progress across layers
- **Color coding**: Different colors represent different computation states
- **Automatic detection**: LED functionality is automatically enabled on Raspberry Pi

### LED Matrix Setup
- Compatible with WS281x LED strips
- Default configuration: 8x4 LED matrix
- GPIO pin 18 (configurable in `led_matrix.py`)

## Development

### Adding New GNN Models
Extend `gnn_splitter.py` to support additional GNN architectures beyond GraphSAGE.

### Custom Datasets
Modify `dataset.py` and implement your own graph dataset loader.

### Tmuxinator
**Tmuxinator** is a tool that allows you to start a tmux session with a complex layout and automatically run commands by configuring a simple yaml configuration file. Tmux is a terminal multiplexer - it can run multiple terminal windows inside a single window. This approach is simpler than opening a new terminal for each command.

1. Moving between terminal panes: hold down `Ctrl` key and use with arrow keys.
1. Switching between tabs: hold down `Shift` and use arrow keys.
1. Killing everything and exiting: press `Ctrl+a`, release, and then press `k`.

Here are some links: [Tmuxinator](https://github.com/tmuxinator/tmuxinator), [Getting starded with Tmux](https://linuxize.com/post/getting-started-with-tmux/), [Tmux Cheat Sheet](https://tmuxcheatsheet.com/)