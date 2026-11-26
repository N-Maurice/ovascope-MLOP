#!/usr/bin/env bash
# Install the system tools OvaScope uses on Ubuntu/Debian:
#   * Docker Engine + Compose plugin (containers)
#   * make (shortcuts in the Makefile)
#   * Node.js 20+ and npm (web interface), only if missing or too old
# and let the current user run docker without sudo. Needs administrator rights:
#
#     sudo bash scripts/install_system_tools.sh
#
# Afterwards log out and back in (or run `newgrp docker`) so the docker group takes effect.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run this script with sudo:  sudo bash scripts/install_system_tools.sh" >&2
  exit 1
fi
TARGET_USER="${SUDO_USER:-$USER}"
# A problem with an unrelated third-party repository (e.g. a renamed PPA) must not stop the install:
# everything below comes from Ubuntu's own archive, whose package lists still refresh.
if ! apt-get update; then
  echo
  echo "Warning: 'apt-get update' reported a problem with a third-party repository (see above)."
  echo "Continuing with Ubuntu's own packages. To fix that repository later, run:"
  echo "  sudo apt-get update --allow-releaseinfo-change"
  echo
fi

# make
command -v make >/dev/null 2>&1 || apt-get install -y make

# Node.js + npm (Ubuntu 24.04+ ships Node 18/20+; the web app needs 20 or newer)
node_major=$(node -v 2>/dev/null | sed 's/v\([0-9]*\).*/\1/' || echo 0)
if [[ -z "$node_major" || "$node_major" -lt 20 ]]; then
  apt-get install -y nodejs npm
fi

# Docker (Ubuntu's own packages: no third-party repository, works on new releases straight away)
if ! (command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1); then
  apt-get install -y docker.io docker-compose-v2 || apt-get install -y docker.io docker-compose-plugin
  systemctl enable --now docker
fi

echo
echo "Installed:"
echo "  make    $(make --version | head -1)"
echo "  node    $(node -v)   npm $(npm -v)"
echo "  docker  $(docker --version)"
echo "  compose $(docker compose version --short 2>/dev/null || docker compose version)"

if ! id -nG "$TARGET_USER" | grep -qw docker; then
  usermod -aG docker "$TARGET_USER"
  echo
  echo "Added '$TARGET_USER' to the docker group. Log out and back in (or run 'newgrp docker'), then check:"
  echo "  docker run --rm hello-world"
fi
