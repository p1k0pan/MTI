# export CUDA_VISIBLE_DEVICES=1,3

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=3,2 \
swift sft \
--model OpenGVLab/InternVL2_5-8B \
--model_type internvl2_5 \
--train_type lora \
--dataset  /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/de_train_500.json \
           /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/en_train_500.json \
           /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/es_train_500.json \
           /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/fr_train_500.json \
           /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/it_train_500.json \
           /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/ja_train_500.json \
           /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/pt_train_500.json \
           /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/zh_train_500.json \
--val_dataset /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/de_val_10.json \
              /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/en_val_10.json \
              /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/es_val_10.json \
              /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/fr_val_10.json \
              /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/it_val_10.json \
              /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/ja_val_10.json \
              /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/pt_val_10.json \
              /ltstorage/home/2pan/dataset/MIT-10M_large/train/mit10_sample1/zh_val_10.json \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--learning_rate 2e-5 \
--lora_rank 16 \
--lora_alpha 32 \
--target_modules all-linear \
--gradient_accumulation_steps 8 \
--eval_steps 250 \
--save_steps 500 \
--lora_dropout 0.1 \
--warmup_ratio 0.1 \
--logging_steps 10 \
--max_length 2046 \
--deepspeed internvl_chat/zero_stage2_config.json \
--dataloader_num_workers 4 \
--output_dir internvl2.5_8b_s500 \
--report_to wandb
