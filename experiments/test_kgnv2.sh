# Date: 10/27/2022

TRAIN_DATA_MODE=$1
TEST_DATA_MODE=$2

SCALE_KPT_COEFF=1


echo " ================================= Test the model trained on ${TRAIN_DATA_MODE} on the ${TEST_DATA_MODE} scenarios =================================== "

cd src

DIST_TH_LIST=(0.01 0.02 0.03)
ANGLE_TH_LIST=(20 30 45)
# the threshold
for (( i=0;i<${#DIST_TH_LIST[@]};i++)); do
    DIST_TH=${DIST_TH_LIST[i]}	
    ANGLE_TH=${ANGLE_TH_LIST[i]}	
    echo "----------- The threshold: Distance_Th = "${DIST_TH}"; Angle_Th = "${ANGLE_TH}" ---------------"
    python test.py grasp_pose \
        --exp_id kgnV2_test_${TRAIN_DATA_MODE}_${TEST_DATA_MODE} \
        --input_mod RGBD \
        --dataset ps_grasp \
        --ps_data_mode ${TEST_DATA_MODE} \
        --keep_res \
        --load_model ../exp/grasp_pose/kgnV2_${TRAIN_DATA_MODE}/model_last.pth \
        --not_prefetch_test \
        --trainval \
        --kpt_type box \
        --pnp_type cvIPPE \
        --center_thresh 0.3 \
        --vis_thresh 0.3 \
        --sep_scale_branch \
        --no_nms \
        --dist_th ${DIST_TH} \
        --angle_th ${ANGLE_TH} \
        --rot_sample_num 30 \
        --trl_sample_num 10 \
        --ori_num 9 \
        --no_kpts_refine \
        --scale_kpts_mode 1 \
        --scale_coeff_k ${SCALE_KPT_COEFF}
        #--debug 5 \
        # --vis_results \
