#!/usr/bin/env bash
# Deploy OvaScope on a fresh Ubuntu VM (tested target: Google Cloud e2-standard-2, Ubuntu 24.04, x86/64).
# Run it on the VM (e.g. in the browser SSH window of Google Cloud):
#
#   curl -fsSL https://raw.githubusercontent.com/N-Maurice/ovascope-MLOP/main/scripts/deploy_vm.sh -o deploy_vm.sh
#   bash deploy_vm.sh
#
# What it does (safe to run again, e.g. after `git push` to update the server):
#   1. installs Docker, Compose and git          4. creates .env with a random admin key (only the first time)
#   2. adds 4 GB of swap (retraining headroom)   5. unpacks ~/ovascope_bundle.tar.gz into data/bundle if present
#   3. clones or updates the repository          6. builds and starts the containers, then prints the addresses
#
# Retraining needs data/bundle (validation set + replay sample, not in git). Create the archive on your computer
#   tar czf ~/ovascope_bundle.tar.gz -C <project> data/bundle
# and upload it to the VM's home folder (browser SSH window: gear/"Upload file"), then run this script again.
set -euo pipefail
REPO=${REPO:-https://github.com/N-Maurice/ovascope-MLOP.git}
DIR=${DIR:-$HOME/ovascope-MLOP}
step() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

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

IP=$(curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip || true)
IP=${IP:-<VM-external-IP>}
cat <<EOF

OvaScope is running.
  Web app : http://$IP:8080      <- create your account NOW: the first account becomes the administrator
  API     : http://$IP:8000      (interactive documentation)
Firewall: ports 8080 and 8000 must be open for this VM (Google Cloud: VPC network > Firewall).
Update later: git push from your computer, then run this script again.
EOF
