#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
scenes=("0001" "0002" "0006" )
conda activate dial
for scene in "${scenes[@]}"; do
    output_path="eval_output/kitti/${scene}_2cam_stage_2_nvs"
    mkdir -p $output_path

    python train_stage_2.py \
        --config configs/kitti_stage_2_nvs.yaml \
        source_path=/data/kitti_pvg/training/image_02/${scene} \
        model_path=$output_path \
        resume=False \
        dynmaic_id_dict_path=eval_output/kitti/${scene}_1cam_stage_1/dynamic_ids.json

done


