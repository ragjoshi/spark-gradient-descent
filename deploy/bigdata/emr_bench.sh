#!/bin/bash
# EMR step, runs on the master node: build the big dataset in S3, then run
# the Spark benchmark on all core nodes and on half of them.
#
#   emr_bench.sh <bucket>
#
# Each run's JSON goes to s3://<bucket>/results/ as soon as it finishes, so
# partial results survive if a later run fails.
set -uxo pipefail
BUCKET=$1
CODE=/home/hadoop/code
SRC=s3://$BUCKET/higgs/HIGGS.csv
DATA=s3://$BUCKET/higgs10x/
COPIES=${COPIES:-10}
ITERS=${ITERS:-20}
PARTS=${PARTS:-128}          # ~4 tasks per core on 32 cores
CORES_PER_NODE=8             # the core fleet is all 8-vCPU, 64 GB types
PY=/opt/bench-venv/bin/python   # from emr_bootstrap.sh
[ -x "$PY" ] || PY=python3

# 1. HIGGS (11M rows, 7.5 GB as CSV) straight from UCI into S3, then
#    server-side copies to make COPIES x HIGGS. The READY marker tells the
#    NumPy instances they can start.
if ! aws s3 ls "$SRC" >/dev/null; then
  curl -fsSL https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz \
    | gunzip | aws s3 cp - "$SRC" || exit 1
fi
aws s3 cp - "s3://$BUCKET/higgs/READY" < /dev/null
for k in $(seq -w 0 $((COPIES - 1))); do
  aws s3 cp --quiet "$SRC" "${DATA}part-$k.csv" &
done
wait

# 2. Spark runs: all nodes, then half, interleaved over reps.
run() {
  local executors=$1 rep=$2
  local name=spark-e$executors-r$rep
  spark-submit --deploy-mode client \
    --num-executors "$executors" --executor-cores $CORES_PER_NODE \
    --executor-memory 34g --conf spark.executor.memoryOverhead=12g \
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.pyspark.python=$PY --conf spark.pyspark.driver.python=$PY \
    --conf spark.scheduler.minRegisteredResourcesRatio=1.0 \
    --conf spark.scheduler.maxRegisteredResourcesWaitingTime=600s \
    --py-files $CODE/train.py,$CODE/gradient.py \
    $CODE/bench.py $((executors * CORES_PER_NODE)) "$rep" --cluster \
    --data "$DATA" --mode blocks --iters "$ITERS" --parts "$PARTS" \
    > /tmp/$name.out 2> /tmp/$name.err
  tail -1 /tmp/$name.out | aws s3 cp - "s3://$BUCKET/results/$name.json"
  aws s3 cp --quiet /tmp/$name.err "s3://$BUCKET/results/logs/$name.err"
}
NODES=$(yarn node -list 2>/dev/null | grep -c RUNNING)
for rep in 0 1; do
  run "$NODES" $rep
  run $((NODES / 2)) $rep
done
