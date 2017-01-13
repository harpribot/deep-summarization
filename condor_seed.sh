#!/bin/sh

# Simple - No Attention - GRU (for seeding)
condorify_gpu_email python train_scripts/train_script_gru_simple_no_attn.py condor_logs/gru_simple_no_attn.txt
