#!/bin/bash
# EMR step, runs on the master node (launched by run.sh): prepare the data
# in S3, then run the Spark benchmark on all core nodes and on half of them.
#
#   emr_bench.sh <bucket> <run name> <source> <label or -> <copies>
#
# <source> is an s3:// file or an s3:// folder ending in "/". The HIGGS
# default (s3://<bucket>/higgs/HIGGS.csv) is downloaded from UCI if missing.
# Each run's JSON goes to s3://<bucket>/results/<run name>/ as soon as it
# finishes, so partial results survive if a later run fails.
set -uxo pipefail
BUCKET=$1
RUN=$2
SRC=$3
LABEL=$4
COPIES=${5:-1}
[ "$LABEL" = "-" ] && LABEL=""
RES=s3://$BUCKET/results/$RUN
CODE=/home/hadoop/code
ITERS=${ITERS:-20}
CORES_PER_NODE=8             # the core fleet is all 8-vCPU, 64 GB types
PY=/opt/bench-venv/bin/python   # from emr_bootstrap.sh
[ -x "$PY" ] || PY=python3

# 1. Data. HIGGS (11M rows, 7.5 GB as CSV) comes straight from UCI into S3.
#    The READY marker tells the NumPy instances they can start.
if [ "$SRC" = "s3://$BUCKET/higgs/HIGGS.csv" ] && ! aws s3 ls "$SRC" >/dev/null; then
  curl -fsSL https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz \
    | gunzip | aws s3 cp - "$SRC" || exit 1
fi
aws s3 cp - "$RES/READY" < /dev/null

# With copies, duplicate every source file COPIES times (server-side copies,
# nothing passes through this machine). A header in each copy is fine:
# train.load_csv drops every header line.
if [ "$COPIES" -gt 1 ]; then
  DATA=s3://$BUCKET/data/$RUN/
  case "$SRC" in
    */) # "aws s3 ls --recursive" prints keys from the bucket root.
        SRC_BUCKET=$(echo "${SRC#s3://}" | cut -d/ -f1)
        OBJECTS=$(aws s3 ls --recursive "$SRC" | awk -v b="s3://$SRC_BUCKET/" '$3 > 0 {print b $4}') ;;
    *) OBJECTS=$SRC ;;
  esac
  for k in $(seq -w 0 $((COPIES - 1))); do
    for obj in $OBJECTS; do
      echo "$obj ${DATA}copy-$k-$(basename "$obj")"
    done
  done | xargs -P 16 -n 2 aws s3 cp --quiet
else
  DATA=$SRC
fi

# 2. Spark runs: all nodes, then half, interleaved over reps. Partitions are
#    fixed at 4 per core of the full cluster, so both sizes do the same tasks.
NODES=$(yarn node -list 2>/dev/null | grep -c RUNNING)
PARTS=${PARTS:-$((NODES * CORES_PER_NODE * 4))}
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
    --data "$DATA" ${LABEL:+--label "$LABEL"} --mode blocks \
    --iters "$ITERS" --parts "$PARTS" \
    > /tmp/$name.out 2> /tmp/$name.err
  tail -1 /tmp/$name.out | aws s3 cp - "$RES/$name.json"
  aws s3 cp --quiet /tmp/$name.err "$RES/logs/$name.err"
}
for rep in 0 1; do
  run "$NODES" $rep
  [ "$NODES" -ge 2 ] && run $((NODES / 2)) $rep
done
