#!/bin/bash
# EC2 user data for the single-machine NumPy baseline (Amazon Linux 2023).
# run.sh fills in the three placeholders below before launching.
#
# The instance builds the same COPIES x HIGGS dataset on its local NVMe disk,
# runs numpy_baseline.py, uploads the result, and shuts down. It is launched
# with "terminate on shutdown", so shutting down deletes it.
BUCKET=__BUCKET__
NAME=__NAME__                # numpy-16gb or numpy-64gb
FLAGS="__FLAGS__"            # e.g. "--stream --iters 6"
COPIES=10

shutdown -h +150             # hard cap: gone after 2.5 hours whatever happens
exec > >(tee /var/log/bench.log) 2>&1
finish() {
  aws s3 cp /var/log/bench.log "s3://$BUCKET/results/logs/$NAME.log" || true
  shutdown -h now
}
trap finish EXIT
set -euxo pipefail

dnf install -y python3-pip
python3 -m pip install --quiet numpy pandas

# Format and mount the instance-store NVMe disk.
DEV=$(lsblk -dpno NAME,MODEL | grep -i "instance storage" | head -1 | awk '{print $1}')
mkfs.xfs -f "$DEV"
mkdir -p /data
mount "$DEV" /data

aws s3 cp --recursive "s3://$BUCKET/code/" /data/code
cd /data/code

# Wait for the EMR step to put HIGGS in S3 (up to 90 minutes).
for _ in $(seq 180); do
  aws s3 ls "s3://$BUCKET/higgs/READY" >/dev/null && break
  sleep 30
done
aws s3 cp "s3://$BUCKET/higgs/HIGGS.csv" /data/HIGGS.csv
python3 make_big_data.py /data/HIGGS.csv /data/big --copies $COPIES
rm /data/HIGGS.csv

# Start from a cold page cache so the first pass is not flattered.
sync
echo 3 > /proc/sys/vm/drop_caches
free -g
python3 numpy_baseline.py --npy /data/big $FLAGS | tee /tmp/out.txt
tail -1 /tmp/out.txt | aws s3 cp - "s3://$BUCKET/results/$NAME.json"
