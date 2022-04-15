
class Encoder :
    def __init__(self, tokenizer, max_input_length, max_target_length, train_flag=False) :
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.train_flag = train_flag

    def __call__(self, examples) :
        bos_token = self.tokenizer.bos_token
        pad_token = self.tokenizer.pad_token

        inputs = [bos_token + doc for doc in examples['context']]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)

        if self.train_flag :
            with self.tokenizer.as_target_tokenizer():
                outputs = [doc + pad_token for doc in examples['summary']]
                labels = self.tokenizer(outputs, max_length=self.max_target_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    