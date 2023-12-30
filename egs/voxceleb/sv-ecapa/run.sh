#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
. ./path.sh || exit 1

stage=3
stop_stage=6

data=/home/jinzezhong/data
exp=/home/jinzezhong/result/jexp
exp_name=joesph_ecapa_tdnn_hubert
gpus="0 1 2 3"

. utils/parse_options.sh || exit 1

exp_dir=$exp/$exp_name

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # In this stage we prepare the raw datasets, including Voxceleb1 and Voxceleb2.
  echo "Stage1: Preparing Voxceleb dataset..."
  ./local/prepare_data.sh --stage 4 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # In this stage we prepare the data index files for training.
  echo "Stage2: Preparing training data index files..."
  python local/prepare_data_csv.py --data_dir $data/vox2_dev
  python local/prepare_data_csv.py --data_dir $data/vox1/dev
  python local/merge_csv.py
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Train the speaker embedding model.
  echo "Stage3: Training the speaker model..."
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  torchrun --nproc_per_node=$num_gpu speakerlab/bin/train_Joseph.py --config conf/ecapa_tdnn_joesph.yaml --gpu $gpus \
           --data $data/vox1_2_concat/train.csv --noise $data/musan/wav.scp --reverb $data/rirs/wav.scp --exp_dir $exp_dir
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Extract embeddings of test datasets.
  echo "Stage4: Extracting speaker embeddings..."
  nj=4
  torchrun --nproc_per_node=$nj --master_port=65534 speakerlab/bin/extract.py --exp_dir $exp_dir \
           --data $data/vox1/test/wav.scp --use_gpu --gpu $gpus
fi

# extract in-domain (FFSVC2020) data to conduct submean backend score.

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Extract embeddings of test datasets.
  echo "Stage5: Extracting speaker embeddings..."
  nj=4
  torchrun --nproc_per_node=$nj speakerlab/bin/extract.py --exp_dir $exp_dir/targets \
           --data $data/FFSVC2020/wav.scp --use_gpu --gpu $gpus
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Output score metrics.
  echo "Stage6: Computing score metrics..."
  trials="$data/vox1/test/trials/vox1_O_cleaned.trial $data/vox1/test/trials/vox1_E_cleaned.trial $data/vox1/test/trials/vox1_H_cleaned.trial"
  # trials="$data/FFSVC2022/trials/trials_dev_keys"
  python speakerlab/bin/compute_score_metrics.py --enrol_data $exp_dir/embeddings --test_data $exp_dir/embeddings \
                                                 --scores_dir $exp_dir/scores --trials $trials
fi
