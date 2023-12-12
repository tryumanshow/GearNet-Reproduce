#!/bin/bash

PREPROCESS_DIR=GearNet/preprocess
DATA_DIR=./downstream/dataset
SAVE_DIR=./downstream/interim

printf "Highly recommend not to execute preprocessing code for EC and GO... \n"
printf "It takes so much time on downloading the dataset. \n"


printf "Preprocessing Enzyme Commision (EC) Prediction Dataset...\n"
printf "It might take some time to download the dataset! \n"
python $PREPROCESS_DIR/preprocess_ft.py --task EC \
      --data_dir $DATA_DIR \
      --save_dir $SAVE_DIR \


printf "Preprocessing Gene Ontology (GO) Prediction Dataset...\n"
printf "It might take some time to download the dataset! \n"
python $PREPROCESS_DIR/preprocess_ft.py --task GO \
      --data_dir $DATA_DIR \
      --save_dir $SAVE_DIR \


printf "Preprocessing Fold Classification (FC) Dataset...\n"
mkdir -p $DATA_DIR/FC
unzip HomologyTAPE.zip -d $DATA_DIR/FC/

python $PREPROCESS_DIR/preprocess_ft.py --task FC \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \


printf "Preprocessing Reaction Classification (RC) Dataset...\n"
mkdir -p $DATA_DIR/RC
unzip ProtFunct.zip -d $DATA_DIR/RC/

python $PREPROCESS_DIR/preprocess_ft.py --task RC \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR