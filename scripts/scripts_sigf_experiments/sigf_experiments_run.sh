#!/usr/bin/bash

# SIGF dataset
DATA=data/retina_datasets/SIGF_time_variant_sequences/test_pairs.csv

# =======
# affine
# =======

# loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method loftr \
--reg_method affine \
--save results/sigf/affine/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method loftr \
--reg_method affine \
--save results/sigf/affine/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method loftr \
--reg_method affine \
--save results/sigf/affine/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method splg \
--reg_method affine \
--save results/sigf/affine/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method splg \
--reg_method affine \
--save results/sigf/affine/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method splg \
--reg_method affine \
--save results/sigf/affine/splg-vm \
--device cuda:0

# ========
# tps-1000
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1000 \
--save results/sigf/tps1000/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1000 \
--save results/sigf/tps1000/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1000 \
--save results/sigf/tps1000/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 1000 \
--save results/sigf/tps1000/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 1000 \
--save results/sigf/tps1000/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 1000 \
--save results/sigf/tps1000/splg-vm \
--device cuda:0

# ========
# tps-100
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 100 \
--save results/sigf/tps100/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 100 \
--save results/sigf/tps100/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 100 \
--save results/sigf/tps100/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 100 \
--save results/sigf/tps100/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 100 \
--save results/sigf/tps100/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 100 \
--save results/sigf/tps100/splg-vm \
--device cuda:0

# ========
# tps-10
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 10 \
--save results/sigf/tps10/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 10 \
--save results/sigf/tps10/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 10 \
--save results/sigf/tps10/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 10 \
--save results/sigf/tps10/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 10 \
--save results/sigf/tps10/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 10 \
--save results/sigf/tps10/splg-vm \
--device cuda:0

# ========
# tps-1
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1 \
--save results/sigf/tps1/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1 \
--save results/sigf/tps1/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 1 \
--save results/sigf/tps1/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 1 \
--save results/sigf/tps1/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 1 \
--save results/sigf/tps1/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 1 \
--save results/sigf/tps1/splg-vm \
--device cuda:0

# ========
# tps-0.1
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.1 \
--save results/sigf/tps0.1/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.1 \
--save results/sigf/tps0.1/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.1 \
--save results/sigf/tps0.1/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.1 \
--save results/sigf/tps0.1/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.1 \
--save results/sigf/tps0.1/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.1 \
--save results/sigf/tps0.1/splg-vm \
--device cuda:0

# ========
# tps-0.01
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.01 \
--save results/sigf/tps0.01/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.01 \
--save results/sigf/tps0.01/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0.01 \
--save results/sigf/tps0.01/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.01 \
--save results/sigf/tps0.01/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.01 \
--save results/sigf/tps0.01/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 0.01 \
--save results/sigf/tps0.01/splg-vm \
--device cuda:0

# ========
# tps-0
# ========

## loftr-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0 \
--save results/sigf/tps0/loftr-g \
--device cuda:0

## loftr-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0 \
--save results/sigf/tps0/loftr-v \
--device cuda:0

## loftr-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method loftr \
--reg_method tps \
--lambda_tps 0 \
--save results/sigf/tps0/loftr-vm \
--device cuda:0

## splg-g
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input img \
--kp_method splg \
--reg_method tps \
--lambda_tps 0 \
--save results/sigf/tps0/splg-g \
--device cuda:0

## splg-v
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input vessel \
--kp_method splg \
--reg_method tps \
--lambda_tps 0 \
--save results/sigf/tps0/splg-v \
--device cuda:0

## splg-vm
python src/pairwise_registrator.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
--input peripheral \
--kp_method splg \
--reg_method tps \
--lambda_tps 0 \
--save results/sigf/tps0/splg-vm \
--device cuda:0