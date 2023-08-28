# LLM_SFT
General SFT scripts for LLMs.

## Environment
We provide [core_requirement.txt](core_requirement.txt) for your convenience.

## Settings
We tested with [vicuna models (v1.3)](https://lmsys.org/blog/2023-03-30-vicuna/) (except llama2-70B) and 10k instructions (padded to max len, file [here](https://github.com/LuJunru/MemoChat/blob/main/data/memochat_instructions/train_10k.json)). Our environment is 900G CPU RAM and 8 x A100 40G GPUs for every computing node. Hyperparameters: Epoch=3, Global Batch=128, Seq Len=2048, Lr=2e-5, Warmup Ratio=0.04, Gen Temperature=0.2.

| Name | Batch | Accum | Total CPU RAM (GB) | Per GPU (GB) | Train Time | Nodes |
| --- | --- | --- | --- | --- | --- | --- |
| T5-3B | 8 | 2 | 73.01 | 37.12 | 1.04h | 1 |
| Vicuna-7B | 16 | 1 | 189.49 | 33.22 | 0.98h | 1 |
| Vicuna-13B | 8 | 2 | 356.42 | 37.29 | 2.35h | 1 |
| Vicuna-33B | 4 | 4 | 790.57 | 38.96 | 5.74h | 1 |
| Llama2-70B-chat-hf | 4 | 2 | 1486.12 | 39.06 | ~36h | 2 |

## Workflow
`RootPath` is the absolute path of this repo.

### Instruction Tuning
Download initial models and put them in [model](model) folder. Put your data in [data](data) folder.
Run `bash code/scripts/tuning.sh RootPath`.

### Inference Testing
1 by 1 simple inference can be found [here](code/codes/eval/get_model_infer_simple.py). This is useful when different sample has different length requirement. You should set a `type` key in your data. We use this format: {'question_id': id, 'text': text, 'type': type}. There's a co-use example in [train script](code/scripts/tuning.sh) as well.
```
python3 code/codes/eval/get_model_infer_simple.py \
    --model-id vicuna-33B \
    --model-path model/vicuna-33B \
    --question-file your-test-data \
    --answer-file your-answer-file-path \
    --num-gpus 8 \
    --ray-num-gpus 2
```

Batch inference can be found [here](code/codes/eval/get_model_infer_batch.py). We use [VLLM](https://github.com/vllm-project/vllm).
```
python3 code/codes/eval/get_model_infer_batch.py \
    --model-path model/vicuna-33B \
    --question-file your-test-data \
    --answer-file your-answer-file-path \
    --max-target-len 512 \
    --num-gpus 8 \
    --num-partitions 2 \
    --temperature 0.8 \
    --top-p 0.95
```
We test batch inference on single node with 1k samples (8~512 tokens) from ShareGPT, asking for maximum 512 tokens.
| model | num-gpus | num-partitions | inference time (1k，8 * A100 40G，greedy search) |
| ---- | ---- | ---- | ---- |
| Vicuna 7B | 8 | 8 | 25s |
| Vicuna 13B | 8 | 8 | 1min08s |
| Vicuna 33B | 8 | 2 | 2min03s |
| Llama2 70B | 8 | 2 | 9min25s |

## Acknowledgement
We thank [Vicuna project](https://github.com/lm-sys/FastChat/tree/main) and [VLLM project](https://github.com/vllm-project/vllm) for their great work.
