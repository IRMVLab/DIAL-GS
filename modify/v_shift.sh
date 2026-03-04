export CUDA_VISIBLE_DEVICES=4

# scenes=("0017085" "0145050" "0147030" "0158150")
scenes=("0017085") #cuda2 waymo1
# scenes=("0147030" "0158150") # cuda3 waymo2



for scene in "${scenes[@]}"; do
    output_path="../eval_output/${scene}_3cam_stage_2_nvs_B"

    mkdir -p $output_path
    
    
   
    
    
    python velocity_shift.py \
        --config ../configs/waymo_modify.yaml \
        source_path=/data/waymo/pvg_scenes/${scene} \
        model_path=$output_path \
        dynmaic_id_dict_path=../eval_output/${scene}_3cam_stage_1/dynamic_ids.json
   
   
    

done
