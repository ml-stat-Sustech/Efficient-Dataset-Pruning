DATASET=$1
if [[ -z $DATASET ]]; then
    echo 'PLEASE INPUT DATASET'
    exit
fi

RATIO=$2
if [[ -z $RATIO ]]; then
    echo 'PLEASE INPUT RATIO'
    exit
fi

GPUs=$3
if [[ -z $GPUs ]]; then
    echo 'PLEASE INPUT GPUs'
    exit
fi

export CUDA_VISIBLE_DEVICES=$GPUs

echo ${DATASET} - ${RATIO}


for principle in lwot-l1-e-i20_f50 lwot-l1-h-i20_f50 ;
do
    echo ${principle}
    python3 finetune.py --seed 1 \
    --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts \
    --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
    --idxs_dir /mnt/sharedata/hdd/jwy/A3C/idxs \
    --dataset $DATASET --pruning --principle ${principle} --ratio $RATIO --pretrain fully --arch resnet18
done