# KGNv1 with additional scale prediction branch, minimum open width rather than canonical width
# AND scaled keypoint coordinates

TRAIN_DATA_MODE=$1      # single or multi
SCALE_KPT_COEFF=1

cd src

python main.py grasp_pose \
    --exp_id kgnV2_${TRAIN_DATA_MODE} \
    --dataset ps_grasp \
    --ps_data_mode ${TRAIN_DATA_MODE} \
    --batch_size 12 \
    --lr 1.25e-4 \
    --load_model ../pretrained_weights/ctdet_coco_dla_2x.pth \
    --gpus 0 \
    --num_workers 4 \
    --num_epochs 400 \
    --lr_step 350,370 \
    --pnp_type cvIPPE \
    --kpt_type box \
    --input_mod RGBD \
    --sep_scale_branch \
    --scale_weight 10 \
    --no_nms \
    --no_kpts_refine \
    --correct_rl \
    --ori_num 9 \
    --scale_kpts_mode 1 \
    --scale_coeff_k ${SCALE_KPT_COEFF}
