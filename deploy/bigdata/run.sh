#!/bin/bash
# Big-data benchmark: Spark on an EMR cluster vs. NumPy on one machine.
#
#   deploy/bigdata/run.sh check     # credentials, region, vCPU quotas; launches nothing
#   deploy/bigdata/run.sh up        # create bucket, upload code, launch everything
#   deploy/bigdata/run.sh status    # what is still running, which results are in
#   deploy/bigdata/run.sh results   # download results and print the comparison
#   deploy/bigdata/run.sh down      # terminate anything still running (keeps S3 data)
#
# What "up" launches (all in your default AWS CLI region):
#   - EMR cluster: 1 on-demand m6i.xlarge master + 4 spot 8-vCPU/64 GB core
#     nodes. Its only step builds 10 x HIGGS in S3 and runs the Spark
#     benchmark; the cluster terminates itself when the step ends (3 h cap).
#   - numpy-16gb: m6id.xlarge (16 GB RAM), reads the data from NVMe every iteration.
#   - numpy-64gb: r6id.2xlarge (64 GB RAM), holds the data in memory.
#   Both NumPy instances terminate themselves when done (2.5 h cap).
#
# Everything is tagged project=spark-gd-bench. S3 data is kept after "down";
# delete the bucket yourself when you have the results (run.sh prints how).
set -euo pipefail

TAG=spark-gd-bench
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HERE=$ROOT/deploy/bigdata

REGION=${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region || true)}}
[ -n "$REGION" ] || { echo "Set a region first: aws configure set region us-east-1"; exit 1; }
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=${BUCKET:-$TAG-$ACCOUNT-$REGION}

quota() {   # quota <code>: current vCPU limit for an EC2 quota
  aws service-quotas get-service-quota --service-code ec2 --quota-code "$1" \
    --query 'Quota.Value' --output text 2>/dev/null || echo "?"
}

check() {
  echo "account $ACCOUNT, region $REGION, bucket s3://$BUCKET"
  local od spot
  od=$(quota L-1216C47A)      # Running On-Demand Standard instances (vCPUs)
  spot=$(quota L-34B43A08)    # All Standard Spot Instance Requests (vCPUs)
  echo "on-demand vCPU limit: $od (need 16: master 4 + numpy-16gb 4 + numpy-64gb 8)"
  echo "spot vCPU limit:      $spot (need 32: 4 core nodes x 8)"
  for v in "$od:16" "$spot:32"; do
    local have=${v%%:*} need=${v##*:}
    if [ "$have" != "?" ] && [ "${have%.*}" -lt "$need" ]; then
      echo "Quota too low. Request an increase in the Service Quotas console (EC2)."
      return 1
    fi
  done
}

up() {
  check
  aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || {
    if [ "$REGION" = us-east-1 ]; then aws s3api create-bucket --bucket "$BUCKET" >/dev/null
    else aws s3api create-bucket --bucket "$BUCKET" \
           --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
  }
  for f in train.py gradient.py bench.py numpy_baseline.py make_big_data.py; do
    aws s3 cp --quiet "$ROOT/$f" "s3://$BUCKET/code/$f"
  done
  aws s3 cp --quiet --recursive "$HERE" "s3://$BUCKET/code/deploy/bigdata/" --exclude "results/*"

  # Creates EMR_DefaultRole and EMR_EC2_DefaultRole if missing (no-op otherwise).
  aws emr create-default-roles >/dev/null

  local release tmp
  release=$(aws emr list-release-labels --filters Prefix=emr-7 \
              --query 'ReleaseLabels[0]' --output text)
  tmp=$(mktemp -d)
  cat > "$tmp/fleets.json" <<EOF
[
  {"Name": "master", "InstanceFleetType": "MASTER", "TargetOnDemandCapacity": 1,
   "InstanceTypeConfigs": [{"InstanceType": "m6i.xlarge"}, {"InstanceType": "m5.xlarge"}]},
  {"Name": "core", "InstanceFleetType": "CORE", "TargetSpotCapacity": 4,
   "InstanceTypeConfigs": [{"InstanceType": "r6i.2xlarge"}, {"InstanceType": "r5.2xlarge"},
                           {"InstanceType": "r7i.2xlarge"}, {"InstanceType": "r6a.2xlarge"}],
   "LaunchSpecifications": {"SpotSpecification": {
     "TimeoutDurationMinutes": 20, "TimeoutAction": "SWITCH_TO_ON_DEMAND",
     "AllocationStrategy": "capacity-optimized"}}}
]
EOF
  cat > "$tmp/steps.json" <<EOF
[{"Type": "CUSTOM_JAR", "Name": "benchmark", "ActionOnFailure": "CONTINUE",
  "Jar": "command-runner.jar",
  "Args": ["bash", "-c", "aws s3 cp --quiet --recursive s3://$BUCKET/code/ /home/hadoop/code && timeout 3h bash /home/hadoop/code/deploy/bigdata/emr_bench.sh $BUCKET"]}]
EOF
  local cluster
  cluster=$(aws emr create-cluster --name "$TAG" --release-label "$release" \
    --applications Name=Spark --use-default-roles \
    --log-uri "s3://$BUCKET/emr-logs/" \
    --instance-fleets "file://$tmp/fleets.json" \
    --bootstrap-actions "Path=s3://$BUCKET/code/deploy/bigdata/emr_bootstrap.sh" \
    --steps "file://$tmp/steps.json" --auto-terminate \
    --tags "project=$TAG" --query ClusterId --output text)
  echo "EMR cluster $cluster ($release) starting"

  launch_numpy numpy-16gb m6id.xlarge "--stream --iters 6"
  launch_numpy numpy-64gb r6id.2xlarge "--iters 20"
  rm -rf "$tmp"
  echo "Launched. Check progress with: deploy/bigdata/run.sh status"
}

launch_numpy() {   # launch_numpy <name> <instance type> <numpy_baseline flags>
  local name=$1 type=$2 flags=$3 userdata id
  userdata=$(mktemp)
  sed -e "s|__BUCKET__|$BUCKET|" -e "s|__NAME__|$name|" -e "s|__FLAGS__|$flags|" \
    "$HERE/numpy_node.sh" > "$userdata"
  # A just-created instance profile can take a few seconds to be usable.
  for attempt in 1 2 3 4 5 6; do
  id=$(aws ec2 run-instances --instance-type "$type" \
    --image-id resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --iam-instance-profile Name=EMR_EC2_DefaultRole \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=20,VolumeType=gp3}' \
    --user-data "file://$userdata" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=project,Value=$TAG},{Key=Name,Value=$name}]" \
    --query 'Instances[0].InstanceId' --output text) && break
  [ "$attempt" = 6 ] && { rm -f "$userdata"; return 1; }
  echo "retrying in 10s..."; sleep 10
  done
  rm -f "$userdata"
  echo "$name: $type $id starting"
}

status() {
  echo "== EMR"
  aws emr list-clusters --active --query "Clusters[?Name=='$TAG'].[Id,Status.State]" --output text
  echo "== NumPy instances"
  aws ec2 describe-instances \
    --filters "Name=tag:project,Values=$TAG" "Name=instance-state-name,Values=pending,running,stopping" \
    --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,InstanceType,State.Name]' --output text
  echo "== Results in s3://$BUCKET/results/"
  aws s3 ls "s3://$BUCKET/results/" 2>/dev/null | grep json || echo "(none yet)"
}

results() {
  local out=$HERE/results
  mkdir -p "$out"
  aws s3 cp --quiet --recursive "s3://$BUCKET/results/" "$out" --exclude "logs/*"
  python3 - "$out" <<'EOF'
import json, os, statistics, sys
from collections import defaultdict
out = sys.argv[1]
runs = defaultdict(list)
for f in sorted(os.listdir(out)):
    if not f.endswith(".json"):
        continue
    try:
        r = json.load(open(os.path.join(out, f)))
    except ValueError:
        print(f"{f}: not valid JSON"); continue
    if "error" in r:
        print(f"{f}: ERROR {r['error']}"); continue
    key = f.rsplit("-r", 1)[0].removesuffix(".json")
    runs[key].append(r)
print(f"{'run':14s} {'rows':>12s} {'s/iter':>8s} {'final loss':>12s}  notes")
for key, rs in sorted(runs.items()):
    s = statistics.median(r["sec_per_iter_warm"] for r in rs)
    note = (f"{rs[0]['cores']} cores, cached {rs[0]['cached_fraction']:.0%}, {len(rs)} reps"
            if "cores" in rs[0] else rs[0].get("source", ""))
    print(f"{key:14s} {rs[0]['records']:>12,} {s:>8.2f} {rs[0]['final_loss']:>12.6f}  {note}")
EOF
}

down() {
  local c
  for c in $(aws emr list-clusters --active --query "Clusters[?Name=='$TAG'].Id" --output text); do
    aws emr terminate-clusters --cluster-ids "$c" && echo "terminating EMR $c"
  done
  local ids
  ids=$(aws ec2 describe-instances \
    --filters "Name=tag:project,Values=$TAG" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text)
  [ -n "$ids" ] && aws ec2 terminate-instances --instance-ids $ids >/dev/null && echo "terminating $ids"
  echo "Compute is shut down. The data in s3://$BUCKET (~85 GB, about \$2/month) is kept."
  echo "Once you have downloaded the results, delete it with:"
  echo "  aws s3 rb s3://$BUCKET --force"
}

case "${1:-}" in
  check|up|status|results|down) "$1" ;;
  *) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
