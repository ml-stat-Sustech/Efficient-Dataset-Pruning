SEED=$1
if [[ -z $SEED ]]; then
    echo 'PLEASE INPUT SEED'
    exit
fi

PRETRAIN=$2
if [[ -z $PRETRAIN ]]; then
    echo 'PLEASE INPUT PRETRAIN'
    exit
fi

ARCH=$3
if [[ -z $ARCH ]]; then
    echo 'PLEASE INPUT ARCH'
    exit
fi

DATASET=$4
if [[ -z $ARCH ]]; then
    echo 'PLEASE INPUT DATASET'
    exit
fi

GPUs=$5
if [[ -z $GPUs ]]; then
    echo 'PLEASE INPUT GPUs'
    exit
fi

export CUDA_VISIBLE_DEVICES=$GPUs


echo ${SEED} - ${PRETRAIN} - ${ARCH} - ${DATASET}

# for s in score score-wot ;
for s in score-wot ;
do
    python3 ${s}.py --seed ${SEED} \
    --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
    --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts \
    --idxs_dir /mnt/sharedata/hdd/jwy/A3C/idxs \
    --dataset ${DATASET} --pretrain ${PRETRAIN} --arch ${ARCH} --interval 0.02 --frequency 50
done
