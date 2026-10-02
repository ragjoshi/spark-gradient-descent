#!/bin/bash
# Big-data benchmark: Spark on an EMR cluster vs. NumPy on one machine.
#
#   deploy/bigdata/run.sh check   [options]  # credentials, quotas, settings; launches nothing
#   deploy/bigdata/run.sh up      [options]  # upload code (and data), launch everything
#   deploy/bigdata/run.sh emr     [options]  # (re)launch the EMR cluster only
#   deploy/bigdata/run.sh numpy   [options] [--only numpy-16gb|numpy-64gb]
#   deploy/bigdata/run.sh status             # what is running, which results are in
#   deploy/bigdata/run.sh results            # download all results and print them
#   deploy/bigdata/run.sh down               # terminate anything still running (keeps S3 data)
#
# Options (the same ones for check, up, emr and numpy):
#   --data SRC    CSV to benchmark: a local file or folder (uploaded to S3), an
#                 s3:// file, or an s3:// folder ending in "/". Default: HIGGS,
#                 downloaded from UCI into S3 by the cluster.
#   --label COL   label column; the files then need a header row. Without it
#                 the label is column 0 and there is no header (HIGGS layout).
#   --copies K    repeat the data K times (default 10 for HIGGS, else 1).
#   --name NAME   results go to results/NAME/ (default: higgs10x, or the file name).
#   --nodes N     Spark core nodes, 8 vCPUs / 64 GB each (default 4).
#
# The data must already be clean: every value numeric, no missing values, a
# 0/1 label. Check a sample in the app first if unsure.
#
# What "up" launches (all in your default AWS CLI region):
#   - EMR cluster: 1 on-demand m6i.xlarge master + N spot core nodes. Its one
#     step prepares the data in S3 and runs the Spark benchmark on N nodes and
#     on N/2; the cluster terminates itself when the step ends (3 h cap).
#   - numpy-16gb: m6id.xlarge (16 GB RAM), re-reads the data from NVMe every iteration.
#   - numpy-64gb: r6id.2xlarge (64 GB RAM), holds the data in memory if it fits.
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
HIGGS_SRC=s3://$BUCKET/higgs/HIGGS.csv

CMD=${1:-}
[ $# -gt 0 ] && shift
DATA="" LABEL="" COPIES="" NAME="" NODES=4 ONLY=all
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --copies) COPIES=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --nodes) NODES=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

# These end up inside shell commands on the cluster, so keep them plain.
for v in "$DATA" "$LABEL" "$NAME"; do
  case "$v" in *[!A-Za-z0-9_./:=+-]*)
    echo "Use only letters, digits and _ . / : = + - in --data, --label and --name: $v"
    exit 1 ;;
  esac
done

if [ -z "$DATA" ]; then
  SRC=$HIGGS_SRC
  NAME=${NAME:-higgs10x}
  COPIES=${COPIES:-10}
  DESCRIPTION="$COPIES copies of HIGGS"
else
  base=$(basename "${DATA%/}")
  NAME=${NAME:-$(echo "${base%%.*}" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '-')}
  COPIES=${COPIES:-1}
  DESCRIPTION="$base"
  [ "$COPIES" -gt 1 ] && DESCRIPTION="$COPIES copies of $base"
  case "$DATA" in
    s3://*) SRC=$DATA ;;
    *) [ -e "$DATA" ] || { echo "No such file or folder: $DATA"; exit 1; }
       if [ -d "$DATA" ]; then SRC=s3://$BUCKET/input/$NAME/
       else SRC=s3://$BUCKET/input/$NAME/$base; fi ;;
  esac
fi
RES=s3://$BUCKET/results/$NAME

quota() {   # quota <code>: current vCPU limit for an EC2 quota
  aws service-quotas get-service-quota --service-code ec2 --quota-code "$1" \
    --query 'Quota.Value' --output text 2>/dev/null || echo "?"
}

check() {
  echo "account $ACCOUNT, region $REGION, bucket s3://$BUCKET"
  echo "run '$NAME': $DESCRIPTION, label ${LABEL:-column 0 (no header)}, $NODES Spark nodes"
  echo "  data: $SRC"
  echo "  results: $RES/"
  local od spot need_spot=$((NODES * 8))
  od=$(quota L-1216C47A)      # Running On-Demand Standard instances (vCPUs)
  spot=$(quota L-34B43A08)    # All Standard Spot Instance Requests (vCPUs)
  echo "on-demand vCPU limit: $od (need 16: master 4 + numpy-16gb 4 + numpy-64gb 8)"
  echo "spot vCPU limit:      $spot (need $need_spot: $NODES core nodes x 8)"
  for v in "$od:16" "$spot:$need_spot"; do
    local have=${v%%:*} need=${v##*:}
    if [ "$have" != "?" ] && [ "${have%.*}" -lt "$need" ]; then
      echo "Quota too low. Request an increase in the Service Quotas console (EC2)."
      return 1
    fi
  done
}

prepare() {   # bucket, code, input data, and the run's meta.json
  aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1 || {
    if [ "$REGION" = us-east-1 ]; then aws s3api create-bucket --bucket "$BUCKET" >/dev/null
    else aws s3api create-bucket --bucket "$BUCKET" \
           --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
  }
  for f in train.py gradient.py bench.py numpy_baseline.py make_big_data.py; do
    aws s3 cp --quiet "$ROOT/$f" "s3://$BUCKET/code/$f"
  done
  aws s3 cp --quiet --recursive "$HERE" "s3://$BUCKET/code/deploy/bigdata/" --exclude "results/*"

  # Local data: sync, so a relaunch does not upload it again.
  if [ -n "$DATA" ] && [ "${DATA#s3://}" = "$DATA" ]; then
    echo "uploading $DATA to s3://$BUCKET/input/$NAME/ ..."
    if [ -d "$DATA" ]; then
      aws s3 sync "$DATA" "s3://$BUCKET/input/$NAME/"
    else
      aws s3 sync "$(dirname "$DATA")" "s3://$BUCKET/input/$NAME/" --exclude "*" --include "$base"
    fi
  fi
  # Data that is already in S3 is ready now; HIGGS is ready once the cluster
  # has downloaded it.
  [ "$SRC" = "$HIGGS_SRC" ] || aws s3 cp - "$RES/READY" < /dev/null

  python3 - "$NAME" "$DESCRIPTION" "$SRC" "$LABEL" "$COPIES" "$NODES" "$REGION" <<'EOF' \
    | aws s3 cp - "$RES/meta.json"
import datetime, json, sys
name, desc, src, label, copies, nodes, region = sys.argv[1:]
print(json.dumps({"name": name, "description": desc, "source": src,
                  "label": label or None, "copies": int(copies), "nodes": int(nodes),
                  "region": region, "date": datetime.date.today().isoformat()}))
EOF
}

up() {
  check
  prepare
  launch_emr
  launch_numpy_all
  echo "Launched. Check progress with: deploy/bigdata/run.sh status"
}

emr() { prepare; launch_emr; }
numpy() { prepare; launch_numpy_all; }

launch_emr() {
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
  {"Name": "core", "InstanceFleetType": "CORE", "TargetSpotCapacity": $NODES,
   "InstanceTypeConfigs": [{"InstanceType": "r6i.2xlarge"}, {"InstanceType": "r5.2xlarge"},
                           {"InstanceType": "r7i.2xlarge"}, {"InstanceType": "r6a.2xlarge"}],
   "LaunchSpecifications": {"SpotSpecification": {
     "TimeoutDurationMinutes": 20, "TimeoutAction": "SWITCH_TO_ON_DEMAND",
     "AllocationStrategy": "capacity-optimized"}}}
]
EOF
  cat > "$tmp/steps.json" <<EOF
[{"Type": "CUSTOM_JAR", "Name": "benchmark-$NAME", "ActionOnFailure": "CONTINUE",
  "Jar": "command-runner.jar",
  "Args": ["bash", "-c", "aws s3 cp --quiet --recursive s3://$BUCKET/code/ /home/hadoop/code && timeout 3h bash /home/hadoop/code/deploy/bigdata/emr_bench.sh $BUCKET $NAME $SRC ${LABEL:--} $COPIES"]}]
EOF
  local cluster
  cluster=$(aws emr create-cluster --name "$TAG" --release-label "$release" \
    --applications Name=Spark --use-default-roles \
    --log-uri "s3://$BUCKET/emr-logs/" \
    --instance-fleets "file://$tmp/fleets.json" \
    --bootstrap-actions "Path=s3://$BUCKET/code/deploy/bigdata/emr_bootstrap.sh" \
    --steps "file://$tmp/steps.json" --auto-terminate \
    --tags "project=$TAG" "run=$NAME" --query ClusterId --output text)
  echo "EMR cluster $cluster ($release, $NODES core nodes) starting"
  rm -rf "$tmp"
}

launch_numpy_all() {
  if [ "$ONLY" = all ] || [ "$ONLY" = numpy-16gb ]; then
    launch_numpy numpy-16gb m6id.xlarge "--stream --iters 6"
  fi
  if [ "$ONLY" = all ] || [ "$ONLY" = numpy-64gb ]; then
    launch_numpy numpy-64gb r6id.2xlarge "--iters 20"
  fi
}

launch_numpy() {   # launch_numpy <name> <instance type> <numpy_baseline flags>
  local name=$1 type=$2 flags=$3 userdata id
  userdata=$(mktemp)
  sed -e "s|__BUCKET__|$BUCKET|" -e "s|__NAME__|$name|" -e "s|__FLAGS__|$flags|" \
      -e "s|__RUN__|$NAME|" -e "s|__SRC__|$SRC|" -e "s|__LABEL__|$LABEL|" \
      -e "s|__COPIES__|$COPIES|" \
    "$HERE/numpy_node.sh" > "$userdata"
  # A just-created instance profile can take a few seconds to be usable.
  for attempt in 1 2 3 4 5 6; do
  id=$(aws ec2 run-instances --instance-type "$type" \
    --image-id resolve:ssm:/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
    --iam-instance-profile Name=EMR_EC2_DefaultRole \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=20,VolumeType=gp3}' \
    --user-data "file://$userdata" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=project,Value=$TAG},{Key=Name,Value=$name},{Key=run,Value=$NAME}]" \
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
    --query 'Reservations[].Instances[].[Tags[?Key==`run`]|[0].Value,Tags[?Key==`Name`]|[0].Value,InstanceType,State.Name]' \
    --output text
  echo "== Results in s3://$BUCKET/results/"
  aws s3 ls --recursive "s3://$BUCKET/results/" 2>/dev/null | grep -v "/logs/" \
    | grep -o 'results/.*\.json' || echo "(none yet)"
}

results() {
  local out=$HERE/results
  mkdir -p "$out"
  aws s3 cp --quiet --recursive "s3://$BUCKET/results/" "$out" --exclude "*/logs/*" --exclude "*/READY"
  python3 - "$out" <<'EOF'
import json, os, statistics, sys
from collections import defaultdict
out = sys.argv[1]
for run in sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))):
    folder = os.path.join(out, run)
    meta = {}
    if os.path.exists(os.path.join(folder, "meta.json")):
        meta = json.load(open(os.path.join(folder, "meta.json")))
    print(f"\n== {run}: {meta.get('description', '')} ({meta.get('date', '')})")
    runs = defaultdict(list)
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".json") or f == "meta.json":
            continue
        try:
            r = json.load(open(os.path.join(folder, f)))
        except ValueError:
            print(f"{f}: not valid JSON"); continue
        if "error" in r:
            print(f"{f}: {r['error']}"); continue
        runs[f.rsplit("-r", 1)[0].removesuffix(".json")].append(r)
    print(f"{'setup':14s} {'rows':>13s} {'s/iter':>8s} {'final loss':>12s}  notes")
    for key, rs in sorted(runs.items()):
        s = statistics.median(r["sec_per_iter_warm"] for r in rs)
        note = (f"{rs[0]['cores']} cores, cached {rs[0]['cached_fraction']:.0%}, {len(rs)} reps"
                if "cores" in rs[0] else rs[0].get("source", ""))
        print(f"{key:14s} {rs[0]['records']:>13,} {s:>8.2f} {rs[0]['final_loss']:>12.6f}  {note}")
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
  echo "Compute is shut down. The data in s3://$BUCKET is kept (about \$0.023 per GB per month)."
  echo "Once you have downloaded the results, delete it with:"
  echo "  aws s3 rb s3://$BUCKET --force"
}

case "$CMD" in
  check|up|emr|numpy|status|results|down) "$CMD" ;;
  *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
