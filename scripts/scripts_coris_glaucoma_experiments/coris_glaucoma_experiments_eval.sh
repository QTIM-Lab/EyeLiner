#!/usr/bin/bash

# =======
# affine
# =======

DATA=results/coris_glaucoma/affine/loftr-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/affine/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/coris_glaucoma/affine/loftr-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/affine/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/coris_glaucoma/affine/loftr-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/affine/loftr-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/coris_glaucoma/affine/splg-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/affine/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/coris_glaucoma/affine/splg-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/affine/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/coris_glaucoma/affine/splg-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/affine/splg-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# =======
# tps1000
# =======

DATA=results/coris_glaucoma/tps1000/loftr-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1000/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/coris_glaucoma/tps1000/loftr-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1000/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/coris_glaucoma/tps1000/loftr-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1000/loftr-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/coris_glaucoma/tps1000/splg-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1000/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/coris_glaucoma/tps1000/splg-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1000/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/coris_glaucoma/tps1000/splg-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1000/splg-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
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

DATA=results/coris_glaucoma/tps100/loftr-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps100/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/coris_glaucoma/tps100/loftr-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps100/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/coris_glaucoma/tps100/loftr-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps100/loftr-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/coris_glaucoma/tps100/splg-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps100/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/coris_glaucoma/tps100/splg-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps100/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/coris_glaucoma/tps100/splg-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps100/splg-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
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

DATA=results/coris_glaucoma/tps10/loftr-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps10/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/coris_glaucoma/tps10/loftr-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps10/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/coris_glaucoma/tps10/loftr-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps10/loftr-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/coris_glaucoma/tps10/splg-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps10/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/coris_glaucoma/tps10/splg-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps10/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/coris_glaucoma/tps10/splg-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps10/splg-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
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

DATA=results/coris_glaucoma/tps1/loftr-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/coris_glaucoma/tps1/loftr-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/coris_glaucoma/tps1/loftr-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1/loftr-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/coris_glaucoma/tps1/splg-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/coris_glaucoma/tps1/splg-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/coris_glaucoma/tps1/splg-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps1/splg-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
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

DATA=results/coris_glaucoma/tps0.01/loftr-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.01/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/coris_glaucoma/tps0.01/loftr-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.01/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/coris_glaucoma/tps0.01/loftr-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.01/loftr-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/coris_glaucoma/tps0.01/splg-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.01/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/coris_glaucoma/tps0.01/splg-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.01/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/coris_glaucoma/tps0.01/splg-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.01/splg-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
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

DATA=results/coris_glaucoma/tps0.1/loftr-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.1/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/coris_glaucoma/tps0.1/loftr-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.1/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/coris_glaucoma/tps0.1/loftr-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.1/loftr-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/coris_glaucoma/tps0.1/splg-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.1/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/coris_glaucoma/tps0.1/splg-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.1/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/coris_glaucoma/tps0.1/splg-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0.1/splg-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
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

DATA=results/coris_glaucoma/tps0/loftr-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/coris_glaucoma/tps0/loftr-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-vm
DATA=results/coris_glaucoma/tps0/loftr-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0/loftr-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/coris_glaucoma/tps0/splg-g/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/coris_glaucoma/tps0/splg-v/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-vm
DATA=results/coris_glaucoma/tps0/splg-vm/test_pairs_new_results.csv
SAVE=results/coris_glaucoma/tps0/splg-vm/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd disk0 \
-md disk1 \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0