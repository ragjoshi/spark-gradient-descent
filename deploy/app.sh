#!/bin/bash
# Host the Streamlit app on one EC2 instance, with no console clicking.
#
#   deploy/app.sh up        # first time: key pair, firewall, instance, deploy
#   deploy/app.sh deploy    # push code changes (or a new APP_PASSWORD) to it
#   deploy/app.sh stop      # pause it: no compute charge while stopped
#   deploy/app.sh start     # resume; prints the new address (it changes)
#   deploy/app.sh allow-ip  # let in the network you are on now (new Wi-Fi)
#   deploy/app.sh public    # let anyone in (set APP_PASSWORD first)
#   deploy/app.sh status    # state and address
#   deploy/app.sh down      # delete the instance and firewall (key pair kept)
#
# The instance is a c7a.2xlarge (8 cores, 16 GB, about $0.41/hour while
# running; about $2.40/month for its disk while stopped). By default only
# your current IP address can open the page. Set APP_PASSWORD in your shell
# before "up" or "deploy" to put a password on the page:
#   APP_PASSWORD=choose-one deploy/app.sh deploy
set -euo pipefail

NAME=spark-gd-app
TYPE=c7a.2xlarge
KEY_FILE=$HOME/.ssh/$NAME.pem
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5)

instance() {   # id of the app instance, if one exists (not terminated)
  aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$NAME" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text | sed 's/^None$//'
}

ip() {
  aws ec2 describe-instances --instance-ids "$1" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

group() {      # id of the security group (firewall), creating it if needed
  local id
  id=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$NAME" \
         --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null | sed 's/^None$//')
  if [ -z "$id" ]; then
    id=$(aws ec2 create-security-group --group-name "$NAME" \
           --description "Spark Scaling Lab app" --query GroupId --output text)
  fi
  echo "$id"
}

allow() {      # allow <cidr>: open ports 22 (deploys) and 80 (the page) to it
  local sg=$1 cidr=$2 port
  for port in 22 80; do
    aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp \
      --port $port --cidr "$cidr" >/dev/null 2>&1 || true   # already allowed
  done
  echo "allowed $cidr"
}

my_ip() { curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]'; }

wait_ready() { # wait until SSH works and the user data has installed Docker
  local host=$1
  echo -n "waiting for $host to finish booting"
  for _ in $(seq 60); do
    if "${SSH[@]}" "ec2-user@$host" 'command -v docker && systemctl is-active --quiet docker' \
         >/dev/null 2>&1; then
      echo " ready"; return 0
    fi
    echo -n "."; sleep 10
  done
  echo; echo "Timed out. Check the instance in the EC2 console."; return 1
}

up() {
  if [ -n "$(instance)" ]; then
    echo "The app instance already exists. Use: deploy/app.sh start, deploy, or status"; exit 1
  fi
  if [ ! -f "$KEY_FILE" ]; then
    mkdir -p "$(dirname "$KEY_FILE")"
    aws ec2 create-key-pair --key-name "$NAME" --query KeyMaterial --output text > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    echo "created SSH key $KEY_FILE"
  fi
  local sg id host
  sg=$(group)
  allow "$sg" "$(my_ip)/32"
  id=$(aws ec2 run-instances --instance-type "$TYPE" \
    --image-id resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --key-name "$NAME" --security-group-ids "$sg" \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}' \
    --user-data "file://$ROOT/deploy/user-data.sh" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=$NAME}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "launched $id ($TYPE)"
  aws ec2 wait instance-running --instance-ids "$id"
  host=$(ip "$id")
  wait_ready "$host"
  "$ROOT/deploy/deploy.sh" "$host" "$KEY_FILE"
  wait_page "$host"
}

wait_page() {  # wait until the app answers, then print the address
  local host=$1
  for _ in $(seq 30); do
    curl -fsS "http://$host/_stcore/health" >/dev/null 2>&1 && break
    sleep 2
  done
  echo "The app is at: http://$host"
}

need() {       # need <state>: id of the instance, which must be in that state
  local id state
  id=$(instance)
  [ -n "$id" ] || { echo "No app instance. Create it with: deploy/app.sh up" >&2; exit 1; }
  state=$(aws ec2 describe-instances --instance-ids "$id" \
            --query 'Reservations[0].Instances[0].State.Name' --output text)
  [ "$state" = "$1" ] || { echo "The instance is $state, not $1." >&2; exit 1; }
  echo "$id"
}

deploy() {
  local id host
  id=$(need running)
  host=$(ip "$id")
  "$ROOT/deploy/deploy.sh" "$host" "$KEY_FILE"
  wait_page "$host"
}

start() {
  local id host
  id=$(need stopped)
  aws ec2 start-instances --instance-ids "$id" >/dev/null
  aws ec2 wait instance-running --instance-ids "$id"
  host=$(ip "$id")
  echo "started $id (the container restarts on its own; give it a minute)"
  wait_page "$host"
}

stop() {
  local id
  id=$(need running)
  aws ec2 stop-instances --instance-ids "$id" >/dev/null
  echo "stopping $id. No compute charge while stopped; start it with: deploy/app.sh start"
}

status() {
  local id
  id=$(instance)
  [ -n "$id" ] || { echo "No app instance."; return; }
  aws ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicIpAddress]' \
    --output text
}

down() {
  local id
  id=$(instance)
  if [ -n "$id" ]; then
    aws ec2 terminate-instances --instance-ids "$id" >/dev/null
    echo "terminating $id"
    aws ec2 wait instance-terminated --instance-ids "$id"
  fi
  local sg
  sg=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$NAME" \
         --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null | sed 's/^None$//')
  [ -n "$sg" ] && aws ec2 delete-security-group --group-id "$sg" && echo "deleted security group $sg"
  echo "Done. The SSH key $KEY_FILE and its AWS key pair are kept for next time."
}

case "${1:-}" in
  up|deploy|start|stop|status|down) "$1" ;;
  allow-ip) allow "$(group)" "$(my_ip)/32" ;;
  public)
    [ -n "${APP_PASSWORD:-}" ] || echo "Warning: no APP_PASSWORD set; anyone with the address can use the app."
    sg=$(group)
    aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp \
      --port 80 --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
    echo "port 80 is open to everyone" ;;
  *) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
