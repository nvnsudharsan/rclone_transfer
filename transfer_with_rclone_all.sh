#!/bin/bash
# transfer_with_rclone.sh
# Multi-variable, multi-year transfer to Levante via rclone

set -u  # error on unset variables

# ============================================================
# Configuration — edit these
# ============================================================
START_YEAR=1989
END_YEAR=1998

# Variable prefixes (filename pattern: <prefix>_YYYY-MM-DD.nc)
VARIABLES=(
    "surface_variables"
    "t_pressure_levels"
    "u_pressure_levels"
    "v_pressure_levels"
    "z_pressure_levels"
    "q_pressure_levels"
    # Add or remove as needed
)

SRC_BASE="base_dir"
DST_BASE="remote_dir"
LOG_DIR="$HOME/rclone_logs"

# rclone tuning
TRANSFERS=16
CHECKERS=16
MULTI_THREAD_STREAMS=4
MULTI_THREAD_CUTOFF="250M"
# ============================================================

mkdir -p "$LOG_DIR"

# Build the full job list so we can show overall progress
TOTAL_JOBS=0
for var in "${VARIABLES[@]}"; do
    for year in $(seq $START_YEAR $END_YEAR); do
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
done

JOB_NUM=0
FAILED_JOBS=()

OVERALL_START=$(date +%s)

for var in "${VARIABLES[@]}"; do
    for year in $(seq $START_YEAR $END_YEAR); do
        JOB_NUM=$((JOB_NUM + 1))

        SRC_DIR="${SRC_BASE}/${year}"
        DST_DIR="${DST_BASE}/graphcast_${year}/"
        LOG_FILE="${LOG_DIR}/${var}_${year}_$(date +%Y%m%d_%H%M%S).log"

        echo ""
        echo "================================================"
        echo "Job $JOB_NUM / $TOTAL_JOBS"
        echo "Variable: $var"
        echo "Year:     $year"
        echo "Started:  $(date)"
        echo "================================================"

        # Skip if source directory doesn't exist
        if [ ! -d "$SRC_DIR" ]; then
            echo "⚠ Source not found: $SRC_DIR — skipping"
            FAILED_JOBS+=("$var/$year (no source dir)")
            continue
        fi

        # Skip if no matching files in source
        if ! ls "${SRC_DIR}"/${var}_*.nc >/dev/null 2>&1; then
            echo "⚠ No ${var}_*.nc files in $SRC_DIR — skipping"
            continue
        fi

        rclone copy \
            "$SRC_DIR/" \
            "$DST_DIR" \
            --include "${var}_*.nc" \
            --transfers "$TRANSFERS" \
            --checkers "$CHECKERS" \
            --multi-thread-streams "$MULTI_THREAD_STREAMS" \
            --multi-thread-cutoff "$MULTI_THREAD_CUTOFF" \
            --no-traverse \
            --progress \
            --stats 30s \
            --retries 10 \
            --low-level-retries 20 \
            --log-file "$LOG_FILE" \
            --log-level INFO

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ $var/$year completed"
        else
            echo "✗ $var/$year FAILED (exit $EXIT_CODE) — see $LOG_FILE"
            FAILED_JOBS+=("$var/$year (exit $EXIT_CODE)")
        fi

        echo "Finished: $(date)"
    done
done

OVERALL_END=$(date +%s)
ELAPSED=$((OVERALL_END - OVERALL_START))

echo ""
echo "================================================"
echo "ALL TRANSFERS DONE."
echo "================================================"
echo "Total wall time: $((ELAPSED / 3600))h $(( (ELAPSED % 3600) / 60 ))m"
echo "Jobs completed: $((JOB_NUM - ${#FAILED_JOBS[@]})) / $TOTAL_JOBS"

if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
    echo ""
    echo "⚠ Failed/skipped jobs (${#FAILED_JOBS[@]}):"
    for j in "${FAILED_JOBS[@]}"; do
        echo "  - $j"
    done
    echo ""
    echo "To retry failures, re-run this script. Completed files will be skipped automatically."
    exit 1
else
    echo "All jobs completed successfully."
fi
