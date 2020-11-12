#! /bin/bash

# quartrain raw
# tools/test_kuro.sh retinaface_quartrain_data+aug_ignore9_nosub1_anchor_lr0.005_qfl_assigner0.35 82 3

# quartrain suffix
# tools/test_kuro.sh --cfg retinaface_quartrain_data+aug_ignore9_nosub1_anchor --epoch epoch_82 --suffix _2150_1600_unlim --gpu 3

# full train with suffix
# tools/test_kuro.sh retinaface_fulltrain_data+aug_ignore9_nosub1_anchor_photo_bfp_regloss2_tstd_minposiou0.3_biupsample_negfocal_rot_dcn 119 8 _2150_1600_rescore_p2x05

# test
# CUDA_VISIBLE_DEVICES=8 python tools/infer_widerface.py configs/widerface/test_time/retina_full_photo_biupsample_ssh_sgdr_gn_diou2_iouaware_2150_1600_nms10k.py work_dirs/retina_full_photo_biupsample_ssh_sgdr_gn_diou2_iouaware/epoch_151.pth data/WIDER_test/images eval_dirs/test/

set -o pipefail

CUDAVD='0'
EVAL=0
PICKLE=0
until [ $# -eq 0 ];do
    key=$1
    case ${key} in
        --cfg)
        WORK_DIR=$2
        shift 2
        ;;
        --epoch)
        BEST_EPOCH=$2
        shift 2
        ;;
        --gpu)
        CUDAVD=$2
        shift 2
        ;;
        --suffix)
        SUFFIX=$2
        shift 2
        ;;
        --eval)
        EVAL=1
        shift
        ;;
        --pkl)
        PICKLE=1
        shift
        ;;
        --tta)
        EXTRA_OPT='--tta-model'
        shift
        ;;
        *)
        echo 'Bad arguments!'
        echo 'Usage:'
        echo 'test_kuro.sh --cfg {cfg_file} --epoch {epoch} --gpu {gpu_id} --suffix {suffix} [--eval --pkl --tta]'
        exit -1
        ;;
    esac
done

mkdir -p eval_dirs/tmp/

if [ ${EVAL} -eq 0 ] && [ ${PICKLE} -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=${CUDAVD} \
    python tools/test_widerface.py \
    configs/widerface/test_time/${WORK_DIR}${SUFFIX}.py \
    work_dirs/${WORK_DIR}/${BEST_EPOCH}.pth \
    --show-dir eval_dirs/tmp/${WORK_DIR}${SUFFIX}${BEST_EPOCH}/ \
    --out eval_dirs/tmp/${WORK_DIR}${SUFFIX}${BEST_EPOCH}.pkl \
    --eval mAP ${EXTRA_OPT} \
    | tee >(sed 's/.*\r//' > eval_dirs/tmp/${WORK_DIR}${SUFFIX}${BEST_EPOCH}.log)
fi

if [ ${PICKLE} -eq 1 ]; then
    python tools/test_widerface_with_result.py \
    configs/widerface/test_time/${WORK_DIR}${SUFFIX}.py \
    eval_dirs/tmp/${WORK_DIR}${SUFFIX}${BEST_EPOCH}.pkl \
    | tee >(sed 's/.*\r//' > eval_dirs/tmp/${WORK_DIR}${SUFFIX}${BEST_EPOCH}.log)
fi

if [ $? -eq 0 ]; then
    echo ''
    echo 'Result txts saved to eval_dirs/tmp/'${WORK_DIR}${SUFFIX}${BEST_EPOCH}
    echo '==============================================================================='
    echo 'Calculating metrics ...'

    python tools/eval_widerface/evaluation.py \
    -p eval_dirs/tmp/${WORK_DIR}${SUFFIX}${BEST_EPOCH} \
    -g /DATA/data/public/WiderFace/eval_tools/ground_truth/ \
    | tee >(sed 's/.*\r//' >> eval_dirs/tmp/${WORK_DIR}${SUFFIX}${BEST_EPOCH}.log)
else
    echo 'Error while executing inference, please check!'
    exit -1
fi

if [ $? -eq 0 ]; then
    echo '==============================================================================='
    echo 'Generating csv file ...'

    python tools/eval_widerface/gen_widerface_csv.py \
    eval_dirs/tmp/${WORK_DIR}${SUFFIX}${BEST_EPOCH}.log \
    eval_dirs/tmp/${WORK_DIR}${SUFFIX}${BEST_EPOCH}.csv
else
    echo 'Error while executing evaluation, please check!'
    exit -1
fi

if [ $? -eq 0 ]; then
    rm -r eval_dirs/tmp/${WORK_DIR}${SUFFIX}${BEST_EPOCH}/
fi
