#!/usr/bin/bash

# =======
# affine
# =======

DATA=results/sigf/affine/loftr-g/test_pairs_results.csv
SAVE=results/sigf/affine/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/sigf/affine/loftr-v/test_pairs_results.csv
SAVE=results/sigf/affine/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/sigf/affine/loftr-vm/test_pairs_results.csv
SAVE=results/sigf/affine/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/sigf/affine/splg-g/test_pairs_results.csv
SAVE=results/sigf/affine/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/sigf/affine/splg-v/test_pairs_results.csv
SAVE=results/sigf/affine/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/sigf/affine/splg-vm/test_pairs_results.csv
SAVE=results/sigf/affine/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# =======
# tps1000
# =======

DATA=results/sigf/tps1000/loftr-g/test_pairs_results.csv
SAVE=results/sigf/tps1000/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/sigf/tps1000/loftr-v/test_pairs_results.csv
SAVE=results/sigf/tps1000/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/sigf/tps1000/loftr-vm/test_pairs_results.csv
SAVE=results/sigf/tps1000/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/sigf/tps1000/splg-g/test_pairs_results.csv
SAVE=results/sigf/tps1000/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/sigf/tps1000/splg-v/test_pairs_results.csv
SAVE=results/sigf/tps1000/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/sigf/tps1000/splg-vm/test_pairs_results.csv
SAVE=results/sigf/tps1000/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# =======
# tps100
# =======

DATA=results/sigf/tps100/loftr-g/test_pairs_results.csv
SAVE=results/sigf/tps100/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/sigf/tps100/loftr-v/test_pairs_results.csv
SAVE=results/sigf/tps100/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/sigf/tps100/loftr-vm/test_pairs_results.csv
SAVE=results/sigf/tps100/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/sigf/tps100/splg-g/test_pairs_results.csv
SAVE=results/sigf/tps100/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/sigf/tps100/splg-v/test_pairs_results.csv
SAVE=results/sigf/tps100/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/sigf/tps100/splg-vm/test_pairs_results.csv
SAVE=results/sigf/tps100/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# =======
# tps10
# =======

DATA=results/sigf/tps10/loftr-g/test_pairs_results.csv
SAVE=results/sigf/tps10/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/sigf/tps10/loftr-v/test_pairs_results.csv
SAVE=results/sigf/tps10/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/sigf/tps10/loftr-vm/test_pairs_results.csv
SAVE=results/sigf/tps10/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/sigf/tps10/splg-g/test_pairs_results.csv
SAVE=results/sigf/tps10/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/sigf/tps10/splg-v/test_pairs_results.csv
SAVE=results/sigf/tps10/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/sigf/tps10/splg-vm/test_pairs_results.csv
SAVE=results/sigf/tps10/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# =======
# tps1
# =======

DATA=results/sigf/tps1/loftr-g/test_pairs_results.csv
SAVE=results/sigf/tps1/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/sigf/tps1/loftr-v/test_pairs_results.csv
SAVE=results/sigf/tps1/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/sigf/tps1/loftr-vm/test_pairs_results.csv
SAVE=results/sigf/tps1/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/sigf/tps1/splg-g/test_pairs_results.csv
SAVE=results/sigf/tps1/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/sigf/tps1/splg-v/test_pairs_results.csv
SAVE=results/sigf/tps1/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/sigf/tps1/splg-vm/test_pairs_results.csv
SAVE=results/sigf/tps1/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# =======
# tps0.01
# =======

DATA=results/sigf/tps0.01/loftr-g/test_pairs_results.csv
SAVE=results/sigf/tps0.01/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/sigf/tps0.01/loftr-v/test_pairs_results.csv
SAVE=results/sigf/tps0.01/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/sigf/tps0.01/loftr-vm/test_pairs_results.csv
SAVE=results/sigf/tps0.01/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/sigf/tps0.01/splg-g/test_pairs_results.csv
SAVE=results/sigf/tps0.01/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/sigf/tps0.01/splg-v/test_pairs_results.csv
SAVE=results/sigf/tps0.01/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/sigf/tps0.01/splg-vm/test_pairs_results.csv
SAVE=results/sigf/tps0.01/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# =======
# tps0.1
# =======

DATA=results/sigf/tps0.1/loftr-g/test_pairs_results.csv
SAVE=results/sigf/tps0.1/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/sigf/tps0.1/loftr-v/test_pairs_results.csv
SAVE=results/sigf/tps0.1/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/sigf/tps0.1/loftr-vm/test_pairs_results.csv
SAVE=results/sigf/tps0.1/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/sigf/tps0.1/splg-g/test_pairs_results.csv
SAVE=results/sigf/tps0.1/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/sigf/tps0.1/splg-v/test_pairs_results.csv
SAVE=results/sigf/tps0.1/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/sigf/tps0.1/splg-vm/test_pairs_results.csv
SAVE=results/sigf/tps0.1/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# =======
# tps0
# =======

DATA=results/sigf/tps0/loftr-g/test_pairs_results.csv
SAVE=results/sigf/tps0/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/sigf/tps0/loftr-v/test_pairs_results.csv
SAVE=results/sigf/tps0/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/sigf/tps0/loftr-vm/test_pairs_results.csv
SAVE=results/sigf/tps0/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/sigf/tps0/splg-g/test_pairs_results.csv
SAVE=results/sigf/tps0/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/sigf/tps0/splg-v/test_pairs_results.csv
SAVE=results/sigf/tps0/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/sigf/tps0/splg-vm/test_pairs_results.csv
SAVE=results/sigf/tps0/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_path \
-mv moving_vessel_path \
-fd fixed_disk_path \
-md moving_disk_path \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0