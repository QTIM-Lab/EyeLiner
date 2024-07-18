#!/usr/bin/bash

# FIRE dataset
DATA=data/retina_datasets/FIRE/fire_time_series.csv

# =======
# affine
# =======

# loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method loftr \
--reg_method affine \
--save results/fire/affine/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method loftr \
--reg_method affine \
--save results/fire/affine/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method loftr \
--reg_method affine \
--save results/fire/affine/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method splg \
--reg_method affine \
--save results/fire/affine/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method splg \
--reg_method affine \
--save results/fire/affine/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method splg \
--reg_method affine \
--save results/fire/affine/splg-vm \
--device cuda:0

# ========
# tps-1000
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1000 \
--save results/fire/tps1000/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1000 \
--save results/fire/tps1000/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1000 \
--save results/fire/tps1000/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 1000 \
--save results/fire/tps1000/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 1000 \
--save results/fire/tps1000/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 1000 \
--save results/fire/tps1000/splg-vm \
--device cuda:0

# ========
# tps-100
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 100 \
--save results/fire/tps100/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 100 \
--save results/fire/tps100/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 100 \
--save results/fire/tps100/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 100 \
--save results/fire/tps100/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 100 \
--save results/fire/tps100/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 100 \
--save results/fire/tps100/splg-vm \
--device cuda:0

# ========
# tps-10
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 10 \
--save results/fire/tps10/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 10 \
--save results/fire/tps10/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 10 \
--save results/fire/tps10/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 10 \
--save results/fire/tps10/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 10 \
--save results/fire/tps10/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 10 \
--save results/fire/tps10/splg-vm \
--device cuda:0

# ========
# tps-1
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1 \
--save results/fire/tps1/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1 \
--save results/fire/tps1/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1 \
--save results/fire/tps1/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 1 \
--save results/fire/tps1/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 1 \
--save results/fire/tps1/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 1 \
--save results/fire/tps1/splg-vm \
--device cuda:0

# ========
# tps-0.1
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.1 \
--save results/fire/tps0.1/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.1 \
--save results/fire/tps0.1/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.1 \
--save results/fire/tps0.1/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.1 \
--save results/fire/tps0.1/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.1 \
--save results/fire/tps0.1/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.1 \
--save results/fire/tps0.1/splg-vm \
--device cuda:0

# ========
# tps-0.01
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.01 \
--save results/fire/tps0.01/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.01 \
--save results/fire/tps0.01/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.01 \
--save results/fire/tps0.01/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.01 \
--save results/fire/tps0.01/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.01 \
--save results/fire/tps0.01/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.01 \
--save results/fire/tps0.01/splg-vm \
--device cuda:0

# ========
# tps-0
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0 \
--save results/fire/tps0/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0 \
--save results/fire/tps0/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0 \
--save results/fire/tps0/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0 \
--save results/fire/tps0/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 0 \
--save results/fire/tps0/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 0 \
--save results/fire/tps0/splg-vm \
--device cuda:0