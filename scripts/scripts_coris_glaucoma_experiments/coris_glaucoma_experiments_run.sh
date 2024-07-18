#!/usr/bin/bash

# CORIS-glaucoma dataset
DATA=data/annotation_guis/coris_glaucoma_reg_paper/test_pairs_new.csv

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
-fd disk0 \
-md disk1 \
--input img \
--kp_method loftr \
--reg_method affine \
--save results/coris_glaucoma/affine/loftr-g \
--device cuda:0

## loftr-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method loftr \
# --reg_method affine \
# --save results/coris_glaucoma/affine/loftr-v \
# --device cuda:0

## loftr-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method loftr \
# --reg_method affine \
# --save results/coris_glaucoma/affine/loftr-vm \
# --device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
--input img \
--kp_method splg \
--reg_method affine \
--save results/coris_glaucoma/affine/splg-g \
--device cuda:0

## splg-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method splg \
# --reg_method affine \
# --save results/coris_glaucoma/affine/splg-v \
# --device cuda:0

## splg-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method splg \
# --reg_method affine \
# --save results/coris_glaucoma/affine/splg-vm \
# --device cuda:0

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
-fd disk0 \
-md disk1 \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1000 \
--save results/coris_glaucoma/tps1000/loftr-g \
--device cuda:0

## loftr-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 1000 \
# --save results/coris_glaucoma/tps1000/loftr-v \
# --device cuda:0

## loftr-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 1000 \
# --save results/coris_glaucoma/tps1000/loftr-vm \
# --device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 1000 \
--save results/coris_glaucoma/tps1000/splg-g \
--device cuda:0

## splg-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 1000 \
# --save results/coris_glaucoma/tps1000/splg-v \
# --device cuda:0

## splg-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 1000 \
# --save results/coris_glaucoma/tps1000/splg-vm \
# --device cuda:0

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
-fd disk0 \
-md disk1 \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 100 \
--save results/coris_glaucoma/tps100/loftr-g \
--device cuda:0

## loftr-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 100 \
# --save results/coris_glaucoma/tps100/loftr-v \
# --device cuda:0

## loftr-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 100 \
# --save results/coris_glaucoma/tps100/loftr-vm \
# --device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 100 \
--save results/coris_glaucoma/tps100/splg-g \
--device cuda:0

## splg-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 100 \
# --save results/coris_glaucoma/tps100/splg-v \
# --device cuda:0

## splg-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 100 \
# --save results/coris_glaucoma/tps100/splg-vm \
# --device cuda:0

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
-fd disk0 \
-md disk1 \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 10 \
--save results/coris_glaucoma/tps10/loftr-g \
--device cuda:0

## loftr-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 10 \
# --save results/coris_glaucoma/tps10/loftr-v \
# --device cuda:0

## loftr-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 10 \
# --save results/coris_glaucoma/tps10/loftr-vm \
# --device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 10 \
--save results/coris_glaucoma/tps10/splg-g \
--device cuda:0

## splg-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 10 \
# --save results/coris_glaucoma/tps10/splg-v \
# --device cuda:0

## splg-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 10 \
# --save results/coris_glaucoma/tps10/splg-vm \
# --device cuda:0

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
-fd disk0 \
-md disk1 \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1 \
--save results/coris_glaucoma/tps1/loftr-g \
--device cuda:0

## loftr-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 1 \
# --save results/coris_glaucoma/tps1/loftr-v \
# --device cuda:0

## loftr-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 1 \
# --save results/coris_glaucoma/tps1/loftr-vm \
# --device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 1 \
--save results/coris_glaucoma/tps1/splg-g \
--device cuda:0

## splg-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 1 \
# --save results/coris_glaucoma/tps1/splg-v \
# --device cuda:0

## splg-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 1 \
# --save results/coris_glaucoma/tps1/splg-vm \
# --device cuda:0

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
-fd disk0 \
-md disk1 \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.1 \
--save results/coris_glaucoma/tps0.1/loftr-g \
--device cuda:0

## loftr-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 0.1 \
# --save results/coris_glaucoma/tps0.1/loftr-v \
# --device cuda:0

## loftr-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 0.1 \
# --save results/coris_glaucoma/tps0.1/loftr-vm \
# --device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.1 \
--save results/coris_glaucoma/tps0.1/splg-g \
--device cuda:0

## splg-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 0.1 \
# --save results/coris_glaucoma/tps0.1/splg-v \
# --device cuda:0

## splg-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 0.1 \
# --save results/coris_glaucoma/tps0.1/splg-vm \
# --device cuda:0

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
-fd disk0 \
-md disk1 \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.01 \
--save results/coris_glaucoma/tps0.01/loftr-g \
--device cuda:0

## loftr-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 0.01 \
# --save results/coris_glaucoma/tps0.01/loftr-v \
# --device cuda:0

## loftr-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 0.01 \
# --save results/coris_glaucoma/tps0.01/loftr-vm \
# --device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.01 \
--save results/coris_glaucoma/tps0.01/splg-g \
--device cuda:0

## splg-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 0.01 \
# --save results/coris_glaucoma/tps0.01/splg-v \
# --device cuda:0

## splg-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 0.01 \
# --save results/coris_glaucoma/tps0.01/splg-vm \
# --device cuda:0

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
-fd disk0 \
-md disk1 \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0 \
--save results/coris_glaucoma/tps0/loftr-g \
--device cuda:0

## loftr-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 0 \
# --save results/coris_glaucoma/tps0/loftr-v \
# --device cuda:0

## loftr-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method loftr \
# --reg_method tps \
# --lambda_tps 0 \
# --save results/coris_glaucoma/tps0/loftr-vm \
# --device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0 \
--save results/coris_glaucoma/tps0/splg-g \
--device cuda:0

# ## splg-v
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input vessel \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 0 \
# --save results/coris_glaucoma/tps0/splg-v \
# --device cuda:0

## splg-vm
# python src/pairwise_registrator.py \
# -d $DATA \
# -f image0 \
# -m image1 \
# -fv vessel0 \
# -mv vessel1 \
# -fd disk0 \
# -md disk1 \
# --input peripheral \
# --kp_method splg \
# --reg_method tps \
# --lambda_tps 0 \
# --save results/coris_glaucoma/tps0/splg-vm \
# --device cuda:0