## Qwen
### 安装

1. 安装qwen2.5vl依赖
```
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]
```

2. 安装llama_factory依赖
```
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install deepspeed wandb
```

### 训练
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

### 推理

1. 推理文件分为 `qwen2vl_w_ocr.py`和`qwen2vl_wo_ocr.py`，分别代表(w)使用ocr结果做推理，和(wo)不使用ocr结果做推理
2. 在推理文件里修改
   - output_folder 输出目录名称
   - root 数据存放的地点
   - args.model_path 模型地址路径
3. `CUDA_VISIBLE_DEVICES=0 python qwen2vl_w_ocr.py`


## InternVL
### 安装
1. 首先进入InterVL文件夹`cd InternVL`，安装InternVL
```
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
mim install mmcv
```
P.S. 如果安装Internvl的环境有问题可参考是否有一下问题
```
1. 默认无痛的环境要求cuda12.4，其他环境或许有办法，依赖过多
2. 装完环境之后，必须先升级bitsandbytes
pip uninstall bitsandbytes
pip install --upgrade "bitsandbytes>=0.43.2"
3. 最后再升级accelerate
pip install --upgrade accelerate
4. cv2手动升级
pip install -U opencv-python
```
2. 安装ms-swift
```
pip install "ms-swift"
pip install wandb deepspeed timm
```

### 训练
1. 在InternVL文件夹下（`cd InternVL`）
2. 更改`mit10_train.sh`里的设置
    - `--dataset`更改每个训练文件的位置
    - `--val-dataset` 更改验证集位置
    - `NPROC_PER_NODE` 使用多少块卡
    - `CUDA_VISIBLE_DEVICES` 指定卡号
    - `--output_dir` 输出位置
3. 开始训练 `bash mit10_train.sh`
4. 训练完成，修改`merge.sh`进行lora权重合并
    - `--adapters` 训练保存的位置
    - `--output_dir` 合并输出的位置
5. 开始合并 `bash merge.sh`

### 推理

1. 推理文件分为 `internvl_w_ocr.py`和`internvl_wo_ocr.py`，分别代表(w)使用ocr结果做推理，和(wo)不使用ocr结果做推理
2. 在推理文件里修改
   - output_folder 输出目录名称
   - root 数据存放的地点
   - model_path 模型地址路径
3. `CUDA_VISIBLE_DEVICES=0 python internvl_w_ocr.py`
