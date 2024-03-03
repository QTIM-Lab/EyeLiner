#!/usr/bin/bash

# UCHealth Inference
DATA=data/retina_datasets/UCHealth_Annotations/grant_images_pairs_wmasks__.csv

# Global 

# loftr-G - Affine
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris/loftr_g --input img --evaluate -l uchealth

# loftr-G - TPS 1000
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps1000/loftr_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 1000

# loftr-G - TPS 100
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps100/loftr_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 100

# loftr-G - TPS 10
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps10/loftr_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 10

# loftr-G - TPS 1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps/loftr_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 1

# loftr-G - TPS 0.1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps0.1/loftr_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 0.1

# loftr-G - TPS 0.01
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps0.01/loftr_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 0.01

# loftr-G - TPS 0
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps0/loftr_g --input img --evaluate -l uchealth --reg_method tps --lambda_tps 0

# Vessel 

# loftr-V - Affine
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris/loftr_v --input structural --evaluate -l uchealth

# loftr-V - TPS 1000
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps1000/loftr_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 1000

# loftr-V - TPS 100
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps100/loftr_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 100

# loftr-V - TPS 10
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps10/loftr_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 10

# loftr-V - TPS 1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps/loftr_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 1

# loftr-V - TPS 0.1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps0.1/loftr_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 0.1

# loftr-V - TPS 0.01
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps0.01/loftr_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 0.01

# loftr-V - TPS 0
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps0/loftr_v --input structural --evaluate -l uchealth --reg_method tps --lambda_tps 0

# Vessel-mask 

# loftr-VM - Affine
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris/loftr_vm --input img --mask structural --evaluate -l uchealth

# loftr-VM - TPS 1000
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps1000/loftr_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 1000

# loftr-VM - TPS 100
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps100/loftr_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 100

# loftr-VM - TPS 10
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps10/loftr_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 10

# loftr-VM - TPS 1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps/loftr_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 1

# loftr-VM - TPS 0.1
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps0.1/loftr_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 0.1

# loftr-VM - TPS 0.01
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps0.01/loftr_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 0.01

# loftr-VM - TPS 0
python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr --save trained_models_new/coris_tps0/loftr_vm --input img --mask structural --evaluate -l uchealth --reg_method tps --lambda_tps 0