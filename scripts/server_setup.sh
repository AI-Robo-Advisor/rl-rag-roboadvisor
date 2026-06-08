#!/usr/bin/env bash
# One-time setup for AWS Lightsail Ubuntu 22.04.
# Run as: ssh -i <key.pem> ubuntu@<SERVER-IP> 'bash -s' < scripts/server_setup.sh
# After running, manually create /opt/roboadvisor/.env (see comments below).

set -euo pipefail

echo "=== Installing Docker ==="
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "=== Adding ubuntu user to docker group ==="
sudo usermod -aG docker ubuntu

echo "=== Creating project directory ==="
sudo mkdir -p /opt/roboadvisor
sudo chown ubuntu:ubuntu /opt/roboadvisor

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
