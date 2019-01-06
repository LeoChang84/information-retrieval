#!/bin/sh

read -p "Please input task type for deletion: " task
if [ ${task} = "a" ];
then
  rm ./data/en_SiameseLSTM_A.h5
elif [ ${task} = 'b' ];
then
  rm ./data/en_SiameseLSTM_B.h5
elif [ ${task} = 'c' ];
then
  rm ./data/en_SiameseLSTM_C.h5
else
  echo "${task}"
fi
rm ./data/history-graph.png
python3 train.py
