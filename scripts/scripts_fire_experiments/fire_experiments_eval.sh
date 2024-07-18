#!/usr/bin/bash

# =======
# affine
# =======

DATA=results/fire/affine/loftr-g/fire_time_series_results.csv
SAVE=results/fire/affine/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-v
DATA=results/fire/affine/loftr-v/fire_time_series_results.csv
SAVE=results/fire/affine/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-vm
DATA=results/fire/affine/loftr-vm/fire_time_series_results.csv
SAVE=results/fire/affine/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-g
DATA=results/fire/affine/splg-g/fire_time_series_results.csv
SAVE=results/fire/affine/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-v
DATA=results/fire/affine/splg-v/fire_time_series_results.csv
SAVE=results/fire/affine/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-vm
DATA=results/fire/affine/splg-vm/fire_time_series_results.csv
SAVE=results/fire/affine/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# =======
# tps1000
# =======

DATA=results/fire/tps1000/loftr-g/fire_time_series_results.csv
SAVE=results/fire/tps1000/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-v
DATA=results/fire/tps1000/loftr-v/fire_time_series_results.csv
SAVE=results/fire/tps1000/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-vm
DATA=results/fire/tps1000/loftr-vm/fire_time_series_results.csv
SAVE=results/fire/tps1000/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-g
DATA=results/fire/tps1000/splg-g/fire_time_series_results.csv
SAVE=results/fire/tps1000/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-v
DATA=results/fire/tps1000/splg-v/fire_time_series_results.csv
SAVE=results/fire/tps1000/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-vm
DATA=results/fire/tps1000/splg-vm/fire_time_series_results.csv
SAVE=results/fire/tps1000/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# =======
# tps100
# =======

DATA=results/fire/tps100/loftr-g/fire_time_series_results.csv
SAVE=results/fire/tps100/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-v
DATA=results/fire/tps100/loftr-v/fire_time_series_results.csv
SAVE=results/fire/tps100/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-vm
DATA=results/fire/tps100/loftr-vm/fire_time_series_results.csv
SAVE=results/fire/tps100/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-g
DATA=results/fire/tps100/splg-g/fire_time_series_results.csv
SAVE=results/fire/tps100/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-v
DATA=results/fire/tps100/splg-v/fire_time_series_results.csv
SAVE=results/fire/tps100/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-vm
DATA=results/fire/tps100/splg-vm/fire_time_series_results.csv
SAVE=results/fire/tps100/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# =======
# tps10
# =======

DATA=results/fire/tps10/loftr-g/fire_time_series_results.csv
SAVE=results/fire/tps10/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-v
DATA=results/fire/tps10/loftr-v/fire_time_series_results.csv
SAVE=results/fire/tps10/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-vm
DATA=results/fire/tps10/loftr-vm/fire_time_series_results.csv
SAVE=results/fire/tps10/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-g
DATA=results/fire/tps10/splg-g/fire_time_series_results.csv
SAVE=results/fire/tps10/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-v
DATA=results/fire/tps10/splg-v/fire_time_series_results.csv
SAVE=results/fire/tps10/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-vm
DATA=results/fire/tps10/splg-vm/fire_time_series_results.csv
SAVE=results/fire/tps10/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# =======
# tps1
# =======

DATA=results/fire/tps1/loftr-g/fire_time_series_results.csv
SAVE=results/fire/tps1/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-v
DATA=results/fire/tps1/loftr-v/fire_time_series_results.csv
SAVE=results/fire/tps1/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-vm
DATA=results/fire/tps1/loftr-vm/fire_time_series_results.csv
SAVE=results/fire/tps1/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-g
DATA=results/fire/tps1/splg-g/fire_time_series_results.csv
SAVE=results/fire/tps1/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-v
DATA=results/fire/tps1/splg-v/fire_time_series_results.csv
SAVE=results/fire/tps1/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-vm
DATA=results/fire/tps1/splg-vm/fire_time_series_results.csv
SAVE=results/fire/tps1/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# =======
# tps0.01
# =======

DATA=results/fire/tps0.01/loftr-g/fire_time_series_results.csv
SAVE=results/fire/tps0.01/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-v
DATA=results/fire/tps0.01/loftr-v/fire_time_series_results.csv
SAVE=results/fire/tps0.01/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-vm
DATA=results/fire/tps0.01/loftr-vm/fire_time_series_results.csv
SAVE=results/fire/tps0.01/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-g
DATA=results/fire/tps0.01/splg-g/fire_time_series_results.csv
SAVE=results/fire/tps0.01/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-v
DATA=results/fire/tps0.01/splg-v/fire_time_series_results.csv
SAVE=results/fire/tps0.01/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-vm
DATA=results/fire/tps0.01/splg-vm/fire_time_series_results.csv
SAVE=results/fire/tps0.01/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# =======
# tps0.1
# =======

DATA=results/fire/tps0.1/loftr-g/fire_time_series_results.csv
SAVE=results/fire/tps0.1/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-v
DATA=results/fire/tps0.1/loftr-v/fire_time_series_results.csv
SAVE=results/fire/tps0.1/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-vm
DATA=results/fire/tps0.1/loftr-vm/fire_time_series_results.csv
SAVE=results/fire/tps0.1/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-g
DATA=results/fire/tps0.1/splg-g/fire_time_series_results.csv
SAVE=results/fire/tps0.1/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-v
DATA=results/fire/tps0.1/splg-v/fire_time_series_results.csv
SAVE=results/fire/tps0.1/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-vm
DATA=results/fire/tps0.1/splg-vm/fire_time_series_results.csv
SAVE=results/fire/tps0.1/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# =======
# tps0
# =======

DATA=results/fire/tps0/loftr-g/fire_time_series_results.csv
SAVE=results/fire/tps0/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-v
DATA=results/fire/tps0/loftr-v/fire_time_series_results.csv
SAVE=results/fire/tps0/loftr-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# loftr-vm
DATA=results/fire/tps0/loftr-vm/fire_time_series_results.csv
SAVE=results/fire/tps0/loftr-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-g
DATA=results/fire/tps0/splg-g/fire_time_series_results.csv
SAVE=results/fire/tps0/splg-g/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-v
DATA=results/fire/tps0/splg-v/fire_time_series_results.csv
SAVE=results/fire/tps0/splg-v/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval

# splg-vm
DATA=results/fire/tps0/splg-vm/fire_time_series_results.csv
SAVE=results/fire/tps0/splg-vm/

python src/eval.py \
-d $DATA \
-f fixed \
-m moving \
-fv fixed_vessel_mask \
-mv moving_vessel_mask \
-fd fixed_disk_mask \
-md moving_disk_mask \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0 \
--fire_eval