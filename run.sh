#!/usr/bin/bash

# FIRE Inference
DATA=data/AMD_multimodal_registration/multimodal_amd_reg_pairwise_wvessels.csv

python src/main.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--reg_method tps \
--lambda_tps 1 \
--save results/ \
--device cuda:0