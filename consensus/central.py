import yaml
import sys
import socket

def send_config_to_node(host, port, config):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    data = yaml.dump(config).encode()
    s.sendall(data)
    s.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: python central.py <config_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    with open(config_file) as f:
        config = yaml.safe_load(f)
    for node_id in config['hosts']:
        host = config['hosts'][node_id]
        port = config['ports'][node_id]
        print(f"Sending config to {node_id} at {host}:{port}")
        try:
            send_config_to_node(host, port, config)
        except Exception as e:
            print(f"Failed to send config to {node_id}: {e}")

if __name__ == "__main__":
    main()
