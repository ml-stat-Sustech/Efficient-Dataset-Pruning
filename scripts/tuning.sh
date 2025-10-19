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
if [[ -z $DATASET ]]; then
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
for ratio in 10 20 30 40 50 60 70 80 90 ;
do
    echo "------ RATIO ${ratio} ------"

    # for split in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 ;
    for split in 45 50 55;
    do
        # specify the range
        # lower_bound=$((${ratio}/2))
        # upper_bound=$((100-${ratio}/2))
        
        # if [ $split -ge $lower_bound ] && [ $split -le $upper_bound ]; then
        #     # for principle in loss lwot ; do
        #     for principle in lwot ; do
        #         echo ${principle}-l1-s${split}-i20_f50
        #         python3 finetune.py --seed ${SEED} \
        #         --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts \
        #         --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
        #         --idxs_dir /mnt/sharedata/hdd/jwy/A3C/idxs \
        #         --dataset $DATASET --pruning --principle ${principle}-l1-s${split}-i20_f50 --ratio $ratio --pretrain $PRETRAIN --arch $ARCH
        #     done
        # fi

        echo lwot-l1-s${split}-i20_f50
        python3 finetune.py --seed ${SEED} \
        --ckpts_dir /mnt/sharedata/hdd/jwy/A3C/ckpts \
        --datasets_dir /mnt/sharedata/hdd/jwy/datasets \
        --idxs_dir /mnt/sharedata/hdd/jwy/A3C/idxs \
        --dataset $DATASET --pruning --principle lwot-l1-s${split}-i20_f50 --ratio $ratio --pretrain $PRETRAIN --arch $ARCH
    done
done
