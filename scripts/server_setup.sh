#!/usr/bin/env bash
# One-time setup for AWS Lightsail Amazon Linux 2023.
# Run as: ssh -i <key.pem> ec2-user@<SERVER-IP> 'bash -s' < scripts/server_setup.sh
# After running, manually create /opt/roboadvisor/.env (see comments below).

set -euo pipefail

echo "=== Installing Docker ==="
sudo dnf update -y
sudo dnf install -y docker

sudo systemctl start docker
sudo systemctl enable docker

echo "=== Installing Docker Compose plugin ==="
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.27.0/docker-compose-linux-x86_64 \
  -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

echo "=== Adding ec2-user to docker group ==="
sudo usermod -aG docker ec2-user

echo "=== Creating project directory ==="
sudo mkdir -p /opt/roboadvisor
sudo chown ec2-user:ec2-user /opt/roboadvisor

echo ""
echo "=== NEXT: create /opt/roboadvisor/.env with these values ==="
cat << 'ENVTEMPLATE'
OPENAI_API_KEY=sk-...
BOK_API_KEY=
CHROMA_PERSIST_DIR=./chroma_db
API_HOST=0.0.0.0
API_PORT=8000
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8501
API_BASE_URL=http://api:8000
LOG_LEVEL=INFO
DOCKERHUB_USERNAME=<your-dockerhub-username>
ENVTEMPLATE
echo ""
echo "Then log out and back in for the docker group to take effect."
echo "=== Server setup complete ==="
