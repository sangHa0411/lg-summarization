from curses import meta
from optparse import Option
from typing import Optional
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments

@dataclass
class ModelArguments : 
    PLM: str = field(
        default="gogamza/kobart-base-v2",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    save_path: str = field(
        default="./checkpoints",
        metadata={
            "help": "Path to save checkpoint from fine tune model"
        },
    )
    
@dataclass
class DataArguments:
    data_path: str = field(
        default="sh110495/lg-summarization",
        metadata={
            "help": "Data path"
        }
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={
            "help": "preprocessing_num_workers for tokenizing"
        }
    )
    max_input_length: int = field(
        default=1024,
        metadata={
            "help": "Max length of input sequence"
        }
    )
    max_target_length: int = field(
        default=512,
        metadata={
            "help": "Max length of target sequence"
        }
    )
    validation_ratio: int = field(
        default=0.2,
        metadata={
            "help": "Validation ratio"
        }
    )
    
@dataclass
class LoggingArguments:
    wandb_path: Optional[str] = field(
        default='wandb.env',
        metadata={"help":'input your dotenv path'},
    )
    wandb_name: Optional[str] = field(
        default='base',
        metadata={"help":'input your dotenv path'},
    )
    group_name: Optional[str] = field(
        default='kobart',
        metadata={"help":'input your dotenv path'},
    )
    project_name: Optional[str] = field(
        default="LG-Summarization",
        metadata={"help": "project name"},
    )