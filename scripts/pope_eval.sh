#!/bin/bash

gpu=$1
export CUDA_VISIBLE_DEVICES=${gpu}

seed=42
models=("llava" "instructblip" "qwen-vl")
datasets=("coco" "aokvqa" "gqa")
types=("random" "popular" "adversarial")


for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for type in "${types[@]}"; do

            case "${model}|${dataset}|${type}" in
                "llava|coco|random")        threshold=0.10; start_layer=30; gamma=5.0 ;;
                "llava|coco|popular")       threshold=0.10; start_layer=27; gamma=3.0 ;;
                "llava|coco|adversarial")   threshold=0.10; start_layer=27; gamma=3.0 ;;

                "llava|aokvqa|random")      threshold=0.10; start_layer=27; gamma=3.0 ;;
                "llava|aokvqa|popular")     threshold=0.10; start_layer=27; gamma=3.0 ;;
                "llava|aokvqa|adversarial") threshold=0.10; start_layer=27; gamma=5.0 ;;

                "llava|gqa|random")         threshold=0.10; start_layer=28; gamma=5.0 ;;
                "llava|gqa|popular")        threshold=0.10; start_layer=28; gamma=5.0 ;;
                "llava|gqa|adversarial")    threshold=0.10; start_layer=28; gamma=5.0 ;;

                "instructblip|coco|random")        threshold=0.10; start_layer=30; gamma=5.0 ;;
                "instructblip|coco|popular")       threshold=0.10; start_layer=26; gamma=5.0 ;;
                "instructblip|coco|adversarial")   threshold=0.10; start_layer=26; gamma=5.0 ;;

                "instructblip|aokvqa|random")      threshold=0.15; start_layer=28; gamma=5.0 ;;
                "instructblip|aokvqa|popular")     threshold=0.15; start_layer=28; gamma=5.0 ;;
                "instructblip|aokvqa|adversarial") threshold=0.15; start_layer=28; gamma=5.0 ;;

                "instructblip|gqa|random")         threshold=0.05; start_layer=21; gamma=5.0 ;;
                "instructblip|gqa|popular")        threshold=0.05; start_layer=21; gamma=5.0 ;;
                "instructblip|gqa|adversarial")    threshold=0.20; start_layer=19; gamma=5.0 ;;

                "qwen-vl|coco|random")        threshold=0.90; start_layer=30; gamma=50.0 ;;
                "qwen-vl|coco|popular")       threshold=0.90; start_layer=30; gamma=10.0 ;;
                "qwen-vl|coco|adversarial")   threshold=0.90; start_layer=28; gamma=5.0 ;;

                "qwen-vl|aokvqa|random")      threshold=0.90; start_layer=25; gamma=50.0 ;;
                "qwen-vl|aokvqa|popular")     threshold=0.90; start_layer=27; gamma=50.0 ;;
                "qwen-vl|aokvqa|adversarial") threshold=0.90; start_layer=23; gamma=3.0 ;;

                "qwen-vl|gqa|random")         threshold=0.90; start_layer=30; gamma=10.0 ;;
                "qwen-vl|gqa|popular")        threshold=0.90; start_layer=19; gamma=5.0 ;;
                "qwen-vl|gqa|adversarial")    threshold=0.90; start_layer=19; gamma=5.0 ;;
                *)
                echo "Unknown: ${model}|${dataset}|${type}"
                exit 1
                ;;
            esac

            echo "Running [pope] evaluation for model: [${model}], dataset: [${dataset}], type: [${type}], threshold=[${threshold}], start_layer=[${start_layer}], gamma=[${gamma}]."

            static_path="/DATA"   # Change your static_path
            pope_path="${static_path}/POPE/${dataset}/${dataset}_pope_${type}.json"
            data_path="${static_path}/$( [ "$dataset" == "gqa" ] && echo "gqa/images" || echo "coco/val2014" )"
            log_path="./logs/pope/${model}/evaluation"
            out_path="./logs/pope/${model}/answer/${dataset}/${type}/cai_${use_cai}_cd_${use_cd}_threshold_${threshold}_layer_${start_layer}_gamma_${gamma}"

            if [ -d "${out_path}" ]; then
                echo "⚠️  Skip: ${out_path} already exists."
                continue
            fi

            python eval_bench/pope/pope_eval_${model}.py \
            --seed ${seed} \
            --model_base ${model} \
            --pope_path ${pope_path} \
            --data_path ${data_path} \
            --log_path ${log_path} \
            --out_path ${out_path} \
            --type ${type} \
            --dataset_name ${dataset} \
            --threshold ${threshold} \
            --start_layer ${start_layer} \
            --gamma ${gamma} \
            --use_cai True \
            --use_cd True
        done
    done
done