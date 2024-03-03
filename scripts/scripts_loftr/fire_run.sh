#!/usr/bin/bash

# FIRE Inference
DATA=data/retina_datasets/FIRE/fire_time_series.csv

# Global 

# SPLG-G - Affine
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire/splg_g --input img --evaluate -l fire

# SPLG-G - TPS 1000
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps1000/splg_g --input img --evaluate -l fire --reg_method tps --lambda_tps 1000

# SPLG-G - TPS 100
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps100/splg_g --input img --evaluate -l fire --reg_method tps --lambda_tps 100

# SPLG-G - TPS 10
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps10/splg_g --input img --evaluate -l fire --reg_method tps --lambda_tps 10

# SPLG-G - TPS 1
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps/splg_g --input img --evaluate -l fire --reg_method tps --lambda_tps 1

# SPLG-G - TPS 0.1
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps0.1/splg_g --input img --evaluate -l fire --reg_method tps --lambda_tps 0.1

# SPLG-G - TPS 0.01
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps0.01/splg_g --input img --evaluate -l fire --reg_method tps --lambda_tps 0.01

# SPLG-G - TPS 0
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps0/splg_g --input img --evaluate -l fire --reg_method tps --lambda_tps 0

# Vessel 

# SPLG-V - Affine
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire/splg_v --input structural --evaluate -l fire

# SPLG-V - TPS 1000
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps1000/splg_v --input structural --evaluate -l fire --reg_method tps --lambda_tps 1000

# SPLG-V - TPS 100
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps100/splg_v --input structural --evaluate -l fire --reg_method tps --lambda_tps 100

# SPLG-V - TPS 10
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps10/splg_v --input structural --evaluate -l fire --reg_method tps --lambda_tps 10

# SPLG-V - TPS 1
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps/splg_v --input structural --evaluate -l fire --reg_method tps --lambda_tps 1

# SPLG-V - TPS 0.1
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps0.1/splg_v --input structural --evaluate -l fire --reg_method tps --lambda_tps 0.1

# SPLG-V - TPS 0.01
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps0.01/splg_v --input structural --evaluate -l fire --reg_method tps --lambda_tps 0.01

# SPLG-V - TPS 0
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps0/splg_v --input structural --evaluate -l fire --reg_method tps --lambda_tps 0

# Vessel-mask 

# SPLG-VM - Affine
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire/splg_vm --input img --mask structural --evaluate -l fire

# SPLG-VM - TPS 1000
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps1000/splg_vm --input img --mask structural --evaluate -l fire --reg_method tps --lambda_tps 1000

# SPLG-VM - TPS 100
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps100/splg_vm --input img --mask structural --evaluate -l fire --reg_method tps --lambda_tps 100

# SPLG-VM - TPS 10
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps10/splg_vm --input img --mask structural --evaluate -l fire --reg_method tps --lambda_tps 10

# SPLG-VM - TPS 1
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps/splg_vm --input img --mask structural --evaluate -l fire --reg_method tps --lambda_tps 1

# SPLG-VM - TPS 0.1
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps0.1/splg_vm --input img --mask structural --evaluate -l fire --reg_method tps --lambda_tps 0.1

# SPLG-VM - TPS 0.01
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps0.01/splg_vm --input img --mask structural --evaluate -l fire --reg_method tps --lambda_tps 0.01

# SPLG-VM - TPS 0
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/fire_tps0/splg_vm --input img --mask structural --evaluate -l fire --reg_method tps --lambda_tps 0