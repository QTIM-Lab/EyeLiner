#!/usr/bin/bash

# SIGF Inference
# DATA=data/retina_datasets/SIGF/SIGF_Annotations/test_pairs_.csv
# SuperPoint+LightGlue
# python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_path -mv moving_vessel_path -fd fixed_disk_path -md moving_disk_path --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/benchmarks/results_superpoint+lightglue_ --input img

# UCHealth Inference
# DATA=data/retina_datasets/UCHealth_10_images/UCHealth_Annotations/grant_images_pairs_wmasks__.csv
# SuperPoint+LightGlue
# python keypoint_registrator.py -d $DATA -f fixed_image -m moving_image -fv fixed_mask -mv moving_mask -fd fixed_disk_path -md moving_disk_path --kp_method loftr --desc_method loftr --match_method loftr -l --save trained_models/uchealth/results_loftr_uchealth_top100 --input img --top_100 #--device "cuda:0"

# FIRE Inference
DATA=data/retina_datasets/FIRE/hard_cases.csv #data/retina_datasets/FIRE/fire_time_series.csv
# SuperPoint+LightGlue
python keypoint_registrator.py -d $DATA -f fixed -m moving -fv fixed_vessel_mask -mv moving_vessel_mask -fd fixed_disk_mask -md moving_disk_mask --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint -l fire --save trained_models/fire/hard_cases_results_superpoint+lightglue_vessels_masked_fire_top100 --input img --mask structural --evaluate --top 100

# AMD Inference
# DATA=data/R21_AFs/pairwise_images_wmasks.csv
# SuperPoint+LightGlue
# python keypoint_registrator.py -d $DATA -f Image1 -m Image2 -fv VesselImage1Path -mv VesselImage2Path -fd None -md None --kp_method superpoint --desc_method superpoint --match_method lightglue_superpoint --save trained_models/benchmarks/results_superpoint+lightglue_vessels_masked_amd_top100 --input img --top_100 --mask vmask