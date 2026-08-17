#!/usr/bin/env bash
set -euo pipefail

project_dir="${1:-/opt/safee-dms}"
if [[ ! -f "${project_dir}/pyproject.toml" ]]; then
  echo "Usage: $0 /absolute/path/to/dms-checkout" >&2
  exit 2
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-picamera2 python3-venv python3-pip ffmpeg libgl1 libglib2.0-0 \
  raspi-utils

python3 -m venv --system-site-packages "${project_dir}/.venv"
"${project_dir}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"${project_dir}/.venv/bin/python" -m pip install -r "${project_dir}/requirements-raspberry.txt"
"${project_dir}/.venv/bin/python" -m pip install --editable "${project_dir}"

echo "Installed Safee DMS. Run the preflight/start command from ${project_dir}:"
echo "  .venv/bin/dms-runtime --config configs/runtime.raspberry_pi5.json"
echo "Install systemd units only after the Pi benchmark and model exports pass."
