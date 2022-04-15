import os
import random
import pandas as pd
from datasets import load_dataset
from utils.encoder import Encoder
from arguments import ModelArguments, DataArguments
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # -- Loading dataset
    dataset = load_dataset(data_args.data_path)
    test_dset = dataset['test']
    id_list = test_dset['id']
    print(test_dset)

    # -- Tokenizing dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.PLM)
    encoder = Encoder(tokenizer=tokenizer, max_input_length=data_args.max_input_length, max_target_length=data_args.max_target_length, train_flag=False)
    test_dset = test_dset.map(encoder, 
        batched=True, 
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=test_dset.column_names,
    )
    
    # -- Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_input_length)
    
    # -- Config & Model
    config = AutoConfig.from_pretrained(model_args.PLM)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.PLM, config=config)

    trainer = Seq2SeqTrainer(model, 
        training_args,
        data_collator=data_collator, 
        tokenizer=tokenizer
    )

    predictions = trainer.predict(test_dataset=test_dset)[0]

    for i, pred in enumerate(predictions) :
        pred_list = pred.tolist()
        if tokenizer.pad_token_id in pred_list :
            index = pred_list.index(tokenizer.pad_token_id)
            predictions[i][index+1:] = tokenizer.pad_token_id

    sentences = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    sentences = [sen.strip() for sen in sentences]

    df = pd.DataFrame({'uid' : id_list, 'summary' : sentences})
    df.to_csv(os.path.join(training_args.output_dir, 'results.csv'), index=False)
    
if __name__ == "__main__" :
    main()