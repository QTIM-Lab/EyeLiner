#!/usr/bin/bash

# FIRE Inference
DATA=results/amd_multimodal_2/tps1000/splg-g/test_pairs_results.csv

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save results/amd_multimodal_2/tps1000/splg-g/ \
--device cuda:0