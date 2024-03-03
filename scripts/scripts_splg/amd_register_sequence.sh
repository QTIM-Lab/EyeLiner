#!/usr/bin/bash

# SPLG-V, TPS-1
DATA=data/AMD_Longitudinal/Long_AMD-GA_CFP_multiple-dates_time_series.csv
SAVE=trained_models_new/amd_coris_sequence/
python keypoint_registrator_sequence.py -d $DATA --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --reg_method tps --lambda_tps 1 --input vmask --save $SAVE