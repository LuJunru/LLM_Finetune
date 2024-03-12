import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import copy
import json

import datasets
from datasets import load_from_disk

import transformers
from transformers import (
    HfArgumentParser,
    T5ForConditionalGeneration, 
    T5Tokenizer,
    T5Config,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessed_path: str = field(
        default=None, metadata={"help": "Path to the preprocessed training data."}
    )

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.cpu().clone().detach()
    return param

def get_peft_state_maybe_zero_3(state_dict, bias):
    if bias == "none":
        to_return = {
            k: state_dict[k].cpu().clone().detach() for k in state_dict if "lora_" in k
        }
    elif bias == "all":
        to_return = {
            k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "llama" in model_args.model_name_or_path.lower():
        from ..flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
    elif "t5" in model_args.model_name_or_path.lower():
        from optimum.bettertransformer import BetterTransformer
    else:
        pass
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    logger.info("start to load dataset")
    train_dataset = load_from_disk(data_args.preprocessed_path)
    column_names = train_dataset.column_names
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    logger.info("load dataset finished")

    if "t5" in model_args.model_name_or_path:
        # load config and tokenziers
        config = T5Config.from_pretrained(model_args.model_name_or_path)
        config.use_cache=False
        # use truncation_side='left' to preserve linking between end of prompt and target labels
        tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path, truncation_side='left')
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        # load config and tokenziers
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config.use_cache = False
        # use truncation_side='left' to preserve linking between end of prompt and target labels
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, truncation_side='left', trust_remote_code=True)
        # initialize modules
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    
    # convert normal model to bettertransformer
    if "t5" in model_args.model_name_or_path.lower():
        model = BetterTransformer.transform(model)

    # Setup seed
    set_seed(training_args.seed)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Setup Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    # Training
    train_result = trainer.train()

    # convert bettertransformer to normal model
    if "t5" in model_args.model_name_or_path.lower():
        trainer.model = BetterTransformer.reverse(trainer.model)
    trainer.save_state()

    # save fp16 model under deepspeed zero2 or zero3
    c_stage = json.load(open(training_args.deepspeed, "r"))["zero_optimization"]["stage"]
    if c_stage in [2, 3]:
        if c_stage == 2:
            w_state_dict = get_peft_state_maybe_zero_3(trainer.model.named_parameters(), "none")
        else:
            w_state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if trainer.is_world_process_zero():
            state_dict = {key: value.half().cpu() for key, value in w_state_dict.items()}
            trainer._save(training_args.output_dir, state_dict=state_dict)
    else:
        trainer.save_model()

if __name__ == "__main__":
    main()
