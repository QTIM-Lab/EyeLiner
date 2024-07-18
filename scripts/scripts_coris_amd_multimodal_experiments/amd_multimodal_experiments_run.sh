#!/usr/bin/bash

# AMD multimodal dataset
DATA=data/annotation_guis/amd_annotation_gui_multimodal/test_pairs_2.csv

# =======
# affine
# =======

# loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method loftr \
--reg_method affine \
--save results/amd_multimodal_2/affine/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method loftr \
--reg_method affine \
--save results/amd_multimodal_2/affine/loftr-v \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method splg \
--reg_method affine \
--save results/amd_multimodal_2/affine/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method splg \
--reg_method affine \
--save results/amd_multimodal_2/affine/splg-v \
--device cuda:0

# ========
# tps-1000
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1000 \
--save results/amd_multimodal_2/tps1000/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1000 \
--save results/amd_multimodal_2/tps1000/loftr-v \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 1000 \
--save results/amd_multimodal_2/tps1000/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 1000 \
--save results/amd_multimodal_2/tps1000/splg-v \
--device cuda:0

# ========
# tps-100
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 100 \
--save results/amd_multimodal_2/tps100/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 100 \
--save results/amd_multimodal_2/tps100/loftr-v \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 100 \
--save results/amd_multimodal_2/tps100/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 100 \
--save results/amd_multimodal_2/tps100/splg-v \
--device cuda:0

# ========
# tps-10
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 10 \
--save results/amd_multimodal_2/tps10/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 10 \
--save results/amd_multimodal_2/tps10/loftr-v \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 10 \
--save results/amd_multimodal_2/tps10/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 10 \
--save results/amd_multimodal_2/tps10/splg-v \
--device cuda:0

# ========
# tps-1
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1 \
--save results/amd_multimodal_2/tps1/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1 \
--save results/amd_multimodal_2/tps1/loftr-v \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 1 \
--save results/amd_multimodal_2/tps1/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 1 \
--save results/amd_multimodal_2/tps1/splg-v \
--device cuda:0

# ========
# tps-0.1
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.1 \
--save results/amd_multimodal_2/tps0.1/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.1 \
--save results/amd_multimodal_2/tps0.1/loftr-v \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.1 \
--save results/amd_multimodal_2/tps0.1/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.1 \
--save results/amd_multimodal_2/tps0.1/splg-v \
--device cuda:0

# ========
# tps-0.01
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.01 \
--save results/amd_multimodal_2/tps0.01/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.01 \
--save results/amd_multimodal_2/tps0.01/loftr-v \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.01 \
--save results/amd_multimodal_2/tps0.01/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.01 \
--save results/amd_multimodal_2/tps0.01/splg-v \
--device cuda:0

# ========
# tps-0
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0 \
--save results/amd_multimodal_2/tps0/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0 \
--save results/amd_multimodal_2/tps0/loftr-v \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0 \
--save results/amd_multimodal_2/tps0/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 0 \
--save results/amd_multimodal_2/tps0/splg-v \
--device cuda:0