#!/bin/bash
# Monitor GPU utilization every 2 seconds and log to file
OUTPUT_FILE="$1"
echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib,power_w,temp_c" > "$OUTPUT_FILE"
while true; do
    TS=$(date +%s)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | while IFS=, read -r util mem_used mem_total power temp; do
        echo "$TS,$util,$mem_used,$mem_total,$power,$temp" >> "$OUTPUT_FILE"
    done
    sleep 2
done
