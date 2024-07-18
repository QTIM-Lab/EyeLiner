#!/usr/bin/bash

# =======
# affine
# =======

DATA=results/amd_multimodal_2/affine/loftr-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/affine/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/amd_multimodal_2/affine/loftr-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/affine/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/amd_multimodal_2/affine/splg-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/affine/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/amd_multimodal_2/affine/splg-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/affine/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# =======
# tps1000
# =======

DATA=results/amd_multimodal_2/tps1000/loftr-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps1000/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/amd_multimodal_2/tps1000/loftr-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps1000/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/amd_multimodal_2/tps1000/splg-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps1000/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 1000 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/amd_multimodal_2/tps1000/splg-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps1000/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
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

DATA=results/amd_multimodal_2/tps100/loftr-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps100/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/amd_multimodal_2/tps100/loftr-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps100/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/amd_multimodal_2/tps100/splg-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps100/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 100 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/amd_multimodal_2/tps100/splg-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps100/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
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

DATA=results/amd_multimodal_2/tps10/loftr-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps10/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/amd_multimodal_2/tps10/loftr-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps10/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/amd_multimodal_2/tps10/splg-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps10/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 10 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/amd_multimodal_2/tps10/splg-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps10/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
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

DATA=results/amd_multimodal_2/tps1/loftr-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps1/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/amd_multimodal_2/tps1/loftr-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps1/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/amd_multimodal_2/tps1/splg-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps1/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/amd_multimodal_2/tps1/splg-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps1/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
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

DATA=results/amd_multimodal_2/tps0.01/loftr-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0.01/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/amd_multimodal_2/tps0.01/loftr-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0.01/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/amd_multimodal_2/tps0.01/splg-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0.01/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 0.01 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/amd_multimodal_2/tps0.01/splg-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0.01/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
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

DATA=results/amd_multimodal_2/tps0.1/loftr-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0.1/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/amd_multimodal_2/tps0.1/loftr-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0.1/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/amd_multimodal_2/tps0.1/splg-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0.1/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 0.1 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/amd_multimodal_2/tps0.1/splg-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0.1/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
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

DATA=results/amd_multimodal_2/tps0/loftr-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0/loftr-g/

# loftr-g
python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# loftr-v
DATA=results/amd_multimodal_2/tps0/loftr-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0/loftr-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-g
DATA=results/amd_multimodal_2/tps0/splg-g/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0/splg-g/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0

# splg-v
DATA=results/amd_multimodal_2/tps0/splg-v/test_pairs_2_results.csv
SAVE=results/amd_multimodal_2/tps0/splg-v/

python src/eval.py \
-d $DATA \
-f image0 \
-m image1 \
-fv vessel0 \
-mv vessel1 \
-fd None \
-md None \
-s 256 \
-r registration_params \
-l 0 \
--detected_keypoints detected_keypoints \
--manual_keypoints landmarks_csv \
--save $SAVE \
--device cuda:0