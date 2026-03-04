#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
scenes=("0017085" "0145050" "0147030" "0158150")
conda activate dial


for scene in "${scenes[@]}"; do
    output_path="eval_output/${scene}_3cam_stage_2"
    mkdir -p $output_path
    python train_stage_2.py \
        --config configs/waymo_stage_2.yaml \
        source_path=/data/waymo/pvg_scenes/${scene} \
        model_path=$output_path \
        dynmaic_id_dict_path=eval_output/${scene}_1cam_stage_1/dynamic_ids.json
done
