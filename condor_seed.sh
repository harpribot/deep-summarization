#!/bin/sh
source ~/.bashrc
# Simple - No Attention - GRU (for seeding)
condorify_gpu_email_hold_on_restart python train_script_gru_simple_no_attn.py condor_logs/gru_simple_no_attn.txt
