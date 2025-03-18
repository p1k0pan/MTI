## 安装

1. 安装qwen2.5vl依赖
```
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]
```

2. 安装llama_factory依赖
```
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install deepspeed
```

## 训练
1. 更新一下[p1k0/MIT10M-refine](https://huggingface.co/datasets/p1k0/MIT10M-refine/tree/main)里面的数据，确保test和train文件夹下载正确
2. `cd LLaMA-Factory`
3. 将sample好的数据放到 `data/`文件夹下
4. 在`data/dataset_info.json` 里面添加训练数据的名称和地址
5. 训练配置文件是 `qwen2vl_lora_sft_mit10.yaml`
    - dataset和eval_dataset的字段是`data/dataset_info.json`里面的名称
    - 注意要修改output_dir字段，决定训练输出的名称
    - deepspeed使用了`examples/deepspeed/ds_z2_config.json`
6. 开始训练: `CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train qwen2vl_lora_sft_mit10.yaml`
7. 训练结束后将lora参数与模型合并，需要修改`examples/merge_lora/qwen2vl_lora_sft.yaml`文件
    - adapter_name_or_path：训练完成的文件夹，对应训练配置文件里的output_dir
    - export_dir：合并的地址
8. 在`LLaMA-Factory`目录下进行合并： `llamafactory-cli export examples/merge_lora/qwen2vl_lora_sft.yaml`

## 推理

1. 在qwen2vl_ocr.py里修改
    - model_name 模型名字，决定了保存的目录名称
    - root 数据存放的地点
    - args.model_path 模型地址路径
2. `CUDA_VISIBLE_DEVICES=0 python qwen2vl_ocr.py`
