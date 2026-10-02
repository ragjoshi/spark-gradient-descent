#!/bin/bash
# EMR bootstrap action: runs on every node before Spark starts.
# Executors import train.py, which needs NumPy and scikit-learn. They go in a
# venv, not the system Python: pip-installing pandas there replaces the
# python-dateutil that the system "aws" command depends on, which breaks it.
# emr_bench.sh points Spark at /opt/bench-venv/bin/python.
set -euxo pipefail
sudo dnf install -y python3-pip
sudo python3 -m venv /opt/bench-venv
sudo /opt/bench-venv/bin/pip install --quiet numpy pandas scikit-learn
