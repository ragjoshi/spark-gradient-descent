#!/bin/bash
# EMR bootstrap action: runs on every node before Spark starts.
# Executors import train.py, which needs NumPy and scikit-learn.
set -euxo pipefail
command -v pip3 >/dev/null || sudo dnf install -y python3-pip
sudo python3 -m pip install --quiet numpy pandas scikit-learn
