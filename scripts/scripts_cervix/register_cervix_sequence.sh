#!/usr/bin/bash

# SPLG-V, TPS-1
DATA=data/cervix_registration.csv
SAVE=trained_models_new/cervix_registration_sequence/
python keypoint_registrator_sequence.py -d $DATA --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --reg_method tps --lambda_tps 1 --input img --save $SAVE