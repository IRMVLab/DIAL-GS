export CUDA_VISIBLE_DEVICES=0

# scenes=("0017085" "0145050" "0147030" "0158150")
scenes=("0017085") 
# scenes=("0147030" "0158150") # cuda3 waymo2



for scene in "${scenes[@]}"; do
    output_path="../eval_output/pixel_exp/${scene}_3cam_stage_2_exp4"
    mkdir -p $output_path
    
    
    python separate.py \
        --config ../configs/waymo_modify.yaml \
        source_path=/data/waymo/pvg_scenes/${scene} \
        model_path=$output_path \
        dynmaic_id_dict_path=../eval_output/${scene}_3cam_stage_1/dynamic_ids.json
   
    

done
