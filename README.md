# LLM_SFT
General SFT scripts for LLMs.

## Environment
We provide [core_requirement.txt](core_requirement.txt) for your convenience.

## Settings
We tested with [vicuna models (v1.3)](https://lmsys.org/blog/2023-03-30-vicuna/) (except Llama-2-70B-chat-hf) and 10k instructions (padded to max len, file [here](https://github.com/LuJunru/MemoChat/blob/main/data/memochat_instructions/train_10k.json)). Our environment is 900G CPU RAM and 8 x A100 40G GPUs for every computing node. Hyperparameters: Epoch=3, Global Batch=128, Seq Len=2048, Lr=2e-5, Warmup Ratio=0.04, Gen Temperature=0.2. We use [BetterTransformer](https://huggingface.co/docs/optimum/bettertransformer/overview) to integrate flash attention, while directly use [Official Version](https://github.com/Dao-AILab/flash-attention) can lead to faster training.

|| T5-3B | Vicuna-7B | Vicuna-13B | Vicuna-33B | Llama2-70B |
| --- | --- | --- | --- | --- | --- |
| Given Batch | 8 | 16 | 8 | 4 | 4 |
| Accumulation | 2 | 1 | 2 | 4 | 2 |
| Nodes | 1 | 1 | 1 | 1 | 2 |
| All CPU RAM | 73.01G | 189.49G | 356.42G | 790.57G | 1486.12G |
| GPU Util | 92.80% | 83.05% | 93.23% | 97.40% | 97.65% |
| SFT Time | 1.04h | 0.98h | 2.35h | 5.74h | 36.67h |
| DeepSpeed | Zero1 | Zero2 + Offload Optimizer | Zero3 + Offload Optimizer | Zero3 + Offload Optimizer & Params | Zero3 + Offload Optimizer |

## Workflow
`RootPath` is the absolute path of this repo.

### Instruction Tuning
Download raw models in [model](model) folder. Put your data in [data](data) folder. Run `bash code/scripts/tuning.sh RootPath`.

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
Batch inference on 1 node with 1k samples (8~512 tokens) from ShareGPT, asking for maximum 512 tokens.
| model | num-gpus | num-partitions | inference time (1k，8 * A100 40G，greedy search) |
| ---- | ---- | ---- | ---- |
| Vicuna 7B | 8 | 8 | 25s |
| Vicuna 13B | 8 | 8 | 1min08s |
| Vicuna 33B | 8 | 2 | 2min03s |
| Llama2 70B | 8 | 2 | 9min25s |

## Acknowledgement
We thank [Vicuna project](https://github.com/lm-sys/FastChat/tree/main) and [VLLM project](https://github.com/vllm-project/vllm) for their great work.
