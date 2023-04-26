# Date: 10/27/2022

TRAIN_DATA_MODE=$1      # single or multi

cd src

python main.py grasp_pose \
    --exp_id kgnV1_${TRAIN_DATA_MODE} \
    --dataset ps_grasp \
    --ps_data_mode ${TRAIN_DATA_MODE} \
    --batch_size 6 \
    --lr 1.25e-4 \
    --load_model ../models/ctdet_coco_dla_2x.pth \
    --gpus 0 \
    --num_workers 4 \
    --num_epochs 400 \
    --lr_step 350,370 \
    --pnp_type cvIPPE \
    --kpt_type box \
    --input_mod RGBD \
    --open_width_canonical 0.1 \
    --no_nms \
    --no_kpts_refine \
    --correct_rl \
    --ori_num 9 