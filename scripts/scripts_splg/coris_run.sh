#!/usr/bin/bash

# UCHealth Inference
DATA=data/retina_datasets/UCHealth_Annotations/grant_images_pairs_wmasks__.csv

# Global 

# SPLG-G - Affine
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris/splg_g --input img --evaluate -l uchealth

# SPLG-G - TPS 1000
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps1000/splg_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 1000

# SPLG-G - TPS 100
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps100/splg_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 100

# SPLG-G - TPS 10
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps10/splg_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 10

# SPLG-G - TPS 1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps/splg_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 1

# SPLG-G - TPS 0.1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps0.1/splg_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 0.1

# SPLG-G - TPS 0.01
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps0.01/splg_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 0.01

# SPLG-G - TPS 0
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps0/splg_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 0

# Vessel 

# SPLG-V - Affine
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris/splg_v --input structural --evaluate -l uchealth

# SPLG-V - TPS 1000
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps1000/splg_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 1000

# SPLG-V - TPS 100
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps100/splg_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 100

# SPLG-V - TPS 10
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps10/splg_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 10

# SPLG-V - TPS 1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps/splg_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 1

# SPLG-V - TPS 0.1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps0.1/splg_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 0.1

# SPLG-V - TPS 0.01
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps0.01/splg_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 0.01

# SPLG-V - TPS 0
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps0/splg_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 0

# Vessel-mask 

# SPLG-VM - Affine
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris/splg_vm --input img --mask structural --evaluate -l uchealth

# SPLG-VM - TPS 1000
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps1000/splg_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 1000

# SPLG-VM - TPS 100
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps100/splg_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 100

# SPLG-VM - TPS 10
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps10/splg_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 10

# SPLG-VM - TPS 1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps/splg_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 1

# SPLG-VM - TPS 0.1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps0.1/splg_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 0.1

# SPLG-VM - TPS 0.01
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps0.01/splg_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 0.01

# SPLG-VM - TPS 0
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models_new/coris_tps0/splg_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 0