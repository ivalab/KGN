cd src

TRAIN_DATA_MODE=single
SCALE_KPT_COEFF=1

python demo/demo.py grasp_pose \
    --demo_data_folder demo/ \
    --exp_id phy_kgnv1 \
    --input_mod RGBD \
    --dataset ps_grasp \
    --keep_res \
    --not_prefetch_test \
    --trainval \
    --kpt_type box \
    --pnp_type cvIPPE \
    --center_thresh 0.30 \
    --vis_thresh 0.30 \
    --no_nms \
    --ori_num 9 \
    --no_kpts_refine \
    --sep_scale_branch \
    --scale_coeff_k ${SCALE_KPT_COEFF} \
    --scale_kpts_mode 1 \
    --load_model ../exp/kgnv2.pth \
    # --debug 2 \
    # --load_model ../exp/grasp_pose/grasp_v2_sKpt${SCALE_KPT_COEFF}_noW_train_${TRAIN_DATA_MODE}_1k/model_last.pth \
