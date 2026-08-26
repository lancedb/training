#!/bin/bash
# usage: gpu_sampler.sh <out.csv>  -- 1 Hz GPU utilization samples until killed
exec nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$1" 2>/dev/null
