#!/usr/bin/bash

# FIRE Inference
DATA=results/multimodal_amd_reg_pairwise_wvessels_results.csv

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-s 256 \
-k None \
-r registration \
--save results/ \
--device cuda:0