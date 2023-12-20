cd ../code
data_dir=/media/result
dataset=CAMS

seed=42
length=4096
output=${dataset}_length_${length}_seed_${seed}
output_dir=${data_dir}/${output}_$(date +%F-%H-%M-%S-%N)
proto_output_dir=${data_dir}/${output}_$(date +%F-%H-%M-%S-%N)_proto
eval_proto_output_dir=eval

CUDA_VISIBLE_DEVICES=0 python3 train.py \
--task_name multilabel \
--dataset_name $dataset \
--output_metrics_filepath /media/result/${output}.json \
--model_dir /media/data/3/lyp/roberta-large \
--seed $seed \
--train_filepath ../data/IntentSDCNL_Training.json \
--dev_filepath ../data/IntentSDCNL_Deving.json \
--output_dir $output_dir \
--proto_output_dir $proto_output_dir \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--proto_dim 128 \
--tau 0.25 \
--learning_rate 2e-5 \
--num_train_epochs 40.0 \
--save_strategy epoch \
--evaluation_strategy epoch \
--metric_for_best_model micro_f1 \
--greater_is_better \
--max_seq_length $length \
--segment_length 64 --do_use_stride --do_use_label_wise_attention

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
--task_name multilabel \
--dataset_name $dataset \
--output_metrics_filepath ../results/1/test/${output}.json \
--model_dir $output_dir \
--test_filepath ../data/IntentSDCNL_Testing.json \
--output_dir $output_dir \
--proto_output_dir $eval_proto_output_dir \
--proto_dim 128 \
--tau 0.25 \
--max_seq_length $length \
--segment_length 64 --do_use_stride --do_use_label_wise_attention


