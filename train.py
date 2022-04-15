
import os
import wandb
import torch
import random
import numpy as np

from dotenv import load_dotenv
from datasets import load_dataset
from utils.encoder import Encoder
from utils.preprocessor import split
from utils.metrics import compute_metrics

from arguments import (ModelArguments, 
    DataArguments, 
    LoggingArguments
)

from functools import partial
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def train():

    # -- Arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()

    # -- Seed
    seed_everything(training_args.seed)

    # -- Data
    dataset = load_dataset(data_args.data_path).shuffle(training_args.seed)
    train_dset = dataset['train']
    print(train_dset)

    # -- Split data
    dset = split(train_dset, data_args.validation_ratio) if training_args.do_eval else train_dset
    column_names = dset['train'].column_names if training_args.do_eval else dset.column_names
    print(dset)

    # -- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.PLM, use_fast=True)

    # -- Preprocessing
    print('\nTokenizing Data')
    encoder = Encoder(tokenizer=tokenizer, max_input_length=data_args.max_input_length, max_target_length=data_args.max_target_length, train_flag=True)
    tokenized_dataset = dset.map(encoder, 
        batched=True, 
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
    )
    print(tokenized_dataset)

    # -- Model
    config = AutoConfig.from_pretrained(model_args.PLM)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.PLM, config=config)

    # -- Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # -- Metrics
    metric_fn = partial(compute_metrics, tokenizer=tokenizer)
  
    # -- Training arguments
    training_args.dataloader_num_workers = data_args.preprocessing_num_workers
    if training_args.do_eval :
        training_args.predict_with_generate = True
        training_args.load_best_model_at_end = True 
        training_args.metric_for_best_model = 'rougeL'

    # -- Trainer
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_dataset['train'] if training_args.do_eval else tokenized_dataset,
        eval_dataset=tokenized_dataset['validation'] if training_args.do_eval else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric_fn
    )

    # -- Wandb
    load_dotenv(dotenv_path=logging_args.wandb_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    wandb.init(
        entity="sangha0411",
        project=logging_args.project_name, 
        name=logging_args.wandb_name,
        group=logging_args.group_name)

    wandb.config.update(training_args)
   
    # -- Training
    if training_args.do_train :
        trainer.train()

        if training_args.do_eval :
            trainer.evaluate()    

        trainer.save_model(model_args.save_path)
    
    wandb.finish()

if __name__ == '__main__' :
    train()

