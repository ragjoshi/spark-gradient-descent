#!/bin/bash
# Copy the project to an EC2 instance and (re)start the app in Docker.
#
#   deploy/deploy.sh <public-ip-or-dns> [path/to/key.pem]
#
# Set APP_PASSWORD in your shell to require a password on the page:
#   APP_PASSWORD=something deploy/deploy.sh 3.91.0.12 ~/.ssh/spark.pem
set -euo pipefail

HOST=${1:?usage: deploy/deploy.sh <host> [key.pem]}
KEY=${2:-}
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[ -n "$KEY" ] && SSH_OPTS+=(-i "$KEY")
REMOTE=ec2-user@$HOST
ROOT=$(cd "$(dirname "$0")/.." && pwd)

rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
  --exclude .git --exclude .venv --exclude __pycache__ \
  --exclude higgs_1m.csv --exclude higgs_1m_upload.csv --exclude HIGGS.csv.gz \
  --exclude .claude \
  "$ROOT/" "$REMOTE:spark-app/"

# Pass the password through stdin, not the command line, so it does not
# show up in the remote process list.
printf '%s' "${APP_PASSWORD:-}" | ssh "${SSH_OPTS[@]}" "$REMOTE" '
  set -euo pipefail
  cat > spark-app.env.tmp
  printf "APP_PASSWORD=%s\n" "$(cat spark-app.env.tmp)" > spark-app.env
  rm spark-app.env.tmp
  chmod 600 spark-app.env
  cd spark-app
  sudo docker build -t spark-app .
  sudo docker rm -f spark-app 2>/dev/null || true
  sudo docker run -d --name spark-app --restart unless-stopped \
    --env-file ../spark-app.env -p 80:8501 spark-app
'
echo "Deployed: http://$HOST"
