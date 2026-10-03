#!/bin/bash
# EC2 user data for Amazon Linux 2023: installs Docker so deploy.sh can
# build and run the app. Paste this into "Advanced details > User data"
# when launching the instance.
set -euxo pipefail
dnf install -y docker rsync
systemctl enable --now docker
usermod -aG docker ec2-user
