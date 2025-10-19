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

GPUs=$4
if [[ -z $GPUs ]]; then
    echo 'PLEASE INPUT GPUs'
    exit
fi

export CUDA_VISIBLE_DEVICES=$GPUs

echo ${PRETRAIN} - ${ARCH}
for DATASET in CXRB10 DeepWeeds DTD FGVCAircraft Sketch;
do
    echo $DATASET
    python3 linear-probing.py --seed ${SEED} \
    --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
    --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts \
    --dataset $DATASET --pretrain $PRETRAIN --arch $ARCH
done