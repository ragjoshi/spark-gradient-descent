#!/bin/bash
# EC2 user data for the single-machine NumPy baseline (Amazon Linux 2023).
# run.sh fills in the placeholders below before launching.
#
# The instance builds the run's dataset on its local NVMe disk (streamed
# from S3, repeated COPIES times), runs numpy_baseline.py, uploads the
# result, and shuts down. It is launched with "terminate on shutdown", so
# shutting down deletes it.
BUCKET=__BUCKET__
NAME=__NAME__                # numpy-16gb or numpy-64gb
FLAGS="__FLAGS__"            # e.g. "--stream --iters 6"
RUN=__RUN__                  # results go to s3://$BUCKET/results/$RUN/
SRC=__SRC__                  # s3:// file or folder ending in /
LABEL="__LABEL__"            # empty: HIGGS layout (no header, label in column 0)
COPIES=__COPIES__
RES=s3://$BUCKET/results/$RUN

shutdown -h +150             # hard cap: gone after 2.5 hours whatever happens
# Also copy output to the serial console, so "aws ec2 get-console-output"
# shows it even if the S3 upload below fails.
exec > >(tee /var/log/bench.log /dev/console) 2>&1
finish() {
  local code=$?
  aws s3 cp /var/log/bench.log "$RES/logs/$NAME.log" || true
  # On failure, stay up 3 minutes so the console output can be read.
  [ "$code" = 0 ] || sleep 180
  shutdown -h now
}
trap finish EXIT
set -euxo pipefail

# A venv, not the system Python: pip-installing pandas there replaces the
# python-dateutil that the system "aws" command depends on, which breaks it.
dnf install -y python3-pip
python3 -m venv /opt/bench
/opt/bench/bin/pip install --quiet numpy pandas
PY=/opt/bench/bin/python

# Format and mount the instance-store NVMe disk.
DEV=$(lsblk -dpno NAME,MODEL | grep -i "instance storage" | head -1 | awk '{print $1}')
mkfs.xfs -f "$DEV"
mkdir -p /data
mount "$DEV" /data

aws s3 cp --recursive "s3://$BUCKET/code/" /data/code
cd /data/code

# Wait until the data is in S3 (the cluster downloads HIGGS; up to 90 minutes).
for _ in $(seq 180); do
  aws s3 ls "$RES/READY" >/dev/null && break
  sleep 30
done
$PY make_big_data.py /data/big "$SRC" ${LABEL:+--label "$LABEL"} --copies "$COPIES"

# Start from a cold page cache so the first pass is not flattered.
sync
echo 3 > /proc/sys/vm/drop_caches
free -g
$PY numpy_baseline.py --npy /data/big $FLAGS | tee /tmp/out.txt
tail -1 /tmp/out.txt | aws s3 cp - "$RES/$NAME.json"
