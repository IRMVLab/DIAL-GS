#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
scenes=("0001" "0002" "0006" )
cams=("02") 
for scene in "${scenes[@]}"; do
    # Stage1.1
    conda activate dial
    output_path="eval_output/${scene}_1cam_stage_1"
    mkdir -p $output_path
    source_path=/data/kitti_pvg/training/image_02/${scene} 
    python train_stage_1_1.py \
        --config configs/kitti_stage_1.yaml \
        source_path=$source_path \
        model_path=$output_path

    echo "Finished warpping for scene ${scene}"

    # Stage1.2
    conda activate boxmot
    for cam in "${cams[@]}"; do
        python train_stage_1_2.py \
            --source $output_path/detection \
            --cam ${cam} \
            --output $output_path/cam${cam}_warp_output \
            --device cuda:0 \
            --reid-weights ./osnet_x0_25_msmt17.pt \
            --all_seq_mask_path $source_path/tracking_data_${cam}/${scene} \
            --start_frame 65 \
            --score-threshold 0.001

        echo "Finished dynamic discovery for scene ${scene} cam ${cam}"
    done
done


