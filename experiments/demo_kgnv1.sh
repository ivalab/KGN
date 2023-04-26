cd src

TRAIN_DATA_MODE=single

python demo/demo.py grasp_pose \
    --demo_data_folder demo/ \
    --exp_id phy_kgnv1 \
    --dataset ps_grasp \
    --input_mod RGD \
    --keep_res \
    --load_model ../exp/kgnv1.pth \
    --pnp_type cvIPPE \
    --kpt_type box \
    --open_width_canonical 0.1 \
    --no_nms \
    --center_thresh 0.3 \
    --kpts_hm_thresh 100 \
    --ori_num 9 \
    --refine_scale \
    # --debug 2
    # --load_model ../exp/grasp_pose/grasp_train_${TRAIN_DATA_MODE}_1k_RL_oriClf9/model_last.pth \