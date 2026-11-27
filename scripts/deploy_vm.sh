#!/usr/bin/env bash
# Deploy OvaScope on a fresh Ubuntu 24.04 VM (x86-64): AWS EC2 (see docs/deployment.md), Google Cloud or any
# other provider. Needs >= 2 vCPU, 8 GB RAM and a 30 GB disk. Run it on the VM, e.g. after `ssh ubuntu@<ip>`:
#
#   curl -fsSL https://raw.githubusercontent.com/N-Maurice/ovascope-MLOP/main/scripts/deploy_vm.sh -o deploy_vm.sh
#   bash deploy_vm.sh                                                  # http://<ip>:8080 and :8000
#   WEB_DOMAIN=ovascope.example.org API_DOMAIN=ovascope.api.example.org bash deploy_vm.sh   # + HTTPS (Caddy)
#
# With domain names, both DNS A records must point to this server and ports 80 + 443 must be open to everyone.
# The names are saved in .env, so later runs (updates) keep HTTPS without repeating them.
#
# What it does (safe to run again, e.g. after `git push` to update the server):
#   1. installs Docker, Compose and git          4. creates .env with a random admin key (only the first time)
#   2. adds 4 GB of swap (retraining headroom)   5. unpacks ~/ovascope_bundle.tar.gz into data/bundle if present
#   3. clones or updates the repository          6. builds and starts the containers, then prints the addresses
#
# Retraining needs data/bundle (validation set + replay sample, not in git). Create the archive on your computer
#   tar czf ~/ovascope_bundle.tar.gz -C <project> data/bundle
# and copy it to the VM's home folder, then run this script (again):
#   scp -i <key.pem> ~/ovascope_bundle.tar.gz ubuntu@<ip>:~          (AWS; Google Cloud: browser SSH "Upload file")
set -euo pipefail
REPO=${REPO:-https://github.com/N-Maurice/ovascope-MLOP.git}
DIR=${DIR:-$HOME/ovascope-MLOP}
step() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

[ "$(uname -m)" = x86_64 ] || { echo "This VM is $(uname -m); tensorflow-cpu needs x86-64 (AWS: not Graviton/t4g)."; exit 1; }
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
if [ ! -d "$DIR/.git" ] && [ "$FREE_GB" -lt 15 ]; then   # first install only
  echo "Only ${FREE_GB} GB free on the disk; the images, swap and data need about 15 GB."
  echo "Enlarge the disk (AWS: EC2 > Volumes > Modify volume, 30 GB) and run this script again."; exit 1
fi

step "1/6 Docker, Compose and git"
if ! command -v docker >/dev/null 2>&1 || ! sudo docker compose version >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y docker.io docker-compose-v2 git
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true     # lets you run `docker` without sudo after the next login

step "2/6 Swap space"
if ! sudo swapon --show | grep -q .; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
free -h | head -2

step "3/6 Code from GitHub"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone "$REPO" "$DIR"
fi
cd "$DIR"

step "4/6 Settings (.env)"
if [ ! -f .env ]; then
  cp .env.example .env
  sed -i "s|^OVASCOPE_ADMIN_KEY=.*|OVASCOPE_ADMIN_KEY=$(openssl rand -hex 24)|" .env
  sed -i "s|^MONGODB_URI=.*|MONGODB_URI=|" .env           # the MongoDB copy stays off unless you set it
  echo "Created .env with a new random admin key (see: grep ADMIN_KEY $DIR/.env)"
fi
for var in HOST_UID HOST_GID; do sed -i "/^$var=/d" .env; done
printf 'HOST_UID=%s\nHOST_GID=%s\n' "$(id -u)" "$(id -g)" >> .env   # the API writes files as this user
WEB_DOMAIN=${WEB_DOMAIN:-$(sed -n 's/^WEB_DOMAIN=//p' .env)}
API_DOMAIN=${API_DOMAIN:-$(sed -n 's/^API_DOMAIN=//p' .env)}
for var in WEB_DOMAIN API_DOMAIN COMPOSE_FILE; do sed -i "/^$var=/d" .env; done
if [ -n "$WEB_DOMAIN" ] && [ -n "$API_DOMAIN" ]; then          # HTTPS through Caddy (docker-compose.cloud.yml)
  printf 'WEB_DOMAIN=%s\nAPI_DOMAIN=%s\nCOMPOSE_FILE=docker-compose.yml:docker-compose.cloud.yml\n' \
    "$WEB_DOMAIN" "$API_DOMAIN" >> .env
  echo "HTTPS enabled for $WEB_DOMAIN and $API_DOMAIN"
elif [ -n "$WEB_DOMAIN$API_DOMAIN" ]; then
  echo "Set both WEB_DOMAIN and API_DOMAIN (or neither)."; exit 1
fi

step "5/6 Data folders"
mkdir -p db data/uploads data/bundle artifacts
if [ -f "$HOME/ovascope_bundle.tar.gz" ]; then
  tar xzf "$HOME/ovascope_bundle.tar.gz" -C "$DIR"
  echo "data/bundle unpacked: $(find data/bundle -type f | wc -l) files (retraining enabled)"
elif [ -z "$(ls -A data/bundle)" ]; then
  echo "No ~/ovascope_bundle.tar.gz yet: prediction works, retraining needs the bundle (see the top of this script)."
fi

step "6/6 Build and start (the first build takes several minutes)"
sudo docker compose up -d --build
echo "Waiting for the API to report healthy ..."
for _ in $(seq 60); do
  curl -sf localhost:8000/health >/dev/null && break
  sleep 5
done
sudo docker compose ps
curl -s localhost:8000/health; echo

# Public address: AWS metadata (IMDSv2), then Google Cloud metadata, then an external echo service.
TOKEN=$(curl -sf -m 2 -X PUT http://169.254.169.254/latest/api/token \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
IP=$(curl -sf -m 2 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4 || true)
[ -n "$IP" ] || IP=$(curl -sf -m 2 -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip || true)
[ -n "$IP" ] || IP=$(curl -sf -m 5 https://checkip.amazonaws.com | tr -d '[:space:]' || true)
IP=${IP:-<VM-public-IP>}
cat <<EOF

OvaScope is running.
  Web app : http://$IP:8080      <- create your account NOW: the first account becomes the administrator
  API     : http://$IP:8000      (interactive documentation)
Firewall: ports 8080 and 8000 must be open for this VM (AWS: security group; Google Cloud: VPC network > Firewall).
Update later: git push from your computer, then run this script again.
EOF
if [ -n "$WEB_DOMAIN" ]; then
  for name in "$WEB_DOMAIN" "$API_DOMAIN"; do
    dns=$(getent ahostsv4 "$name" | awk 'NR==1{print $1}' || true)
    if [ "$dns" = "$IP" ]; then state="OK"; else state="points to '${dns:-nothing}', expected $IP: fix the DNS A record"; fi
    echo "  https://$name   (DNS: $state)"
  done
  echo "HTTPS needs ports 80 and 443 open to 0.0.0.0/0; certificate status: sudo docker compose logs caddy"
fi
