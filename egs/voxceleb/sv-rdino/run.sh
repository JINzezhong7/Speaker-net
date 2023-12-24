#!/bin/bash


set -e
. ./path.sh || exit 1

stage=3
stop_stage=4

data=/home9a/zezhong/data
exp=exp
# exp_name=
gpus="0 1"

. utils/parse_options.sh || exit 1

exp_dir=$exp

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage1: Preparing Voxceleb dataset ..."
  ./local/prepare_data_rdino.sh --stage 1 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage2: Training the speaker model ..."
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  torchrun --nproc_per_node=$num_gpu speakerlab/bin/train_rdino.py --config conf/rdino.yaml --gpu $gpus \
           --data $data/vox2_dev/wav.scp --noise $data/musan/wav.scp --reverb $data/rirs/wav.scp --exp_dir $exp_dir
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage4: Extracting speaker embeddings ..."
  nj=4
  torchrun --nproc_per_node=$nj speakerlab/bin/extract_rdino.py --exp_dir $exp_dir \
           --data $data/vox1/test/wav.scp --use_gpu --gpu $gpus
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage5: Computing score metrics ..."
  trials="$data/vox1/test/trials/vox1_O_cleaned.trial"
  python speakerlab/bin/compute_score_metrics.py --enrol_data $exp_dir/embeddings --test_data $exp_dir/embeddings \
                                                 --scores_dir $exp_dir/scores --trials $trials --p_target 0.05
fi
