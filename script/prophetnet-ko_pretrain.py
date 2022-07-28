import numpy as np
import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from transformers import ProphetNetConfig, ProphetNetTokenizer, ProphetNetForConditionalGeneration
from transformers import XLMProphetNetConfig, XLMProphetNetTokenizer, XLMProphetNetForConditionalGeneration




## Specifiy which model to train (base/large)
model_size = "base"


## Data collator for ProphetNet
class ProphetNetDataCollator(DataCollatorForSeq2Seq):
        
    def __call__(self, batch):
        input_ids_batch = []
        input_ids_len = []
        
        decoder_input_ids_batch = []
        decoder_input_ids_len = []
        
        labels_batch = []
        labels_len = []
        
        labels_ngram_batch = []
        labels_ngram_len = []
        
        for example in batch:
            input_ids = example["input_ids"]
            input_ids_len.append(len(input_ids))
            input_ids_batch.append(input_ids)
            
            decoder_input_ids = example["decoder_input_ids"]
            decoder_input_ids_len.append(len(decoder_input_ids))
            decoder_input_ids_batch.append(decoder_input_ids)
            
            labels = example["labels"]
            labels_len.append(len(labels))
            labels_batch.append(labels)
            
            labels_ngram = example["labels_ngram"]
            labels_ngram_len.append(len(labels_ngram))
            labels_ngram_batch.append(labels_ngram)

        input_ids_padded = self.process_encoded_text(input_ids_batch, input_ids_len, self.tokenizer.pad_token_id)
        decoder_input_ids_padded = self.process_encoded_text(decoder_input_ids_batch, decoder_input_ids_len, self.tokenizer.pad_token_id)
        labels_padded = self.process_encoded_text(labels_batch, labels_len, self.label_pad_token_id)
        labels_ngram_padded = self.process_encoded_text(labels_ngram_batch, labels_ngram_len, self.label_pad_token_id)
        
        attention_mask = self.generate_attention_mask(input_ids_padded, self.tokenizer.pad_token_id)
        decoder_attention_mask = self.generate_attention_mask(decoder_input_ids_padded, self.tokenizer.pad_token_id)
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids_padded,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels_padded,
            "labels_ngram": labels_ngram_padded
        }

    def process_encoded_text(self, sequences, sequences_len, pad_token_id):
        sequences_max_len = np.max(sequences_len)
        max_length = min(sequences_max_len, self.max_length)
        padded_sequences = self.pad_sequences(sequences, max_length, pad_token_id)
        return torch.LongTensor(padded_sequences)

    def generate_attention_mask(self, input_ids, pad_token_id):
        return (input_ids != pad_token_id).long()
    
    def pad_sequences(self, sequences, max_length, pad_token_id):
        num_samples = len(sequences)
        padded_sequences = np.full((num_samples, max_length), pad_token_id)
        for i, sequence in enumerate(sequences):
            sequence = np.array(sequence)[:max_length]
            padded_sequences[i, :len(sequence)] = sequence
        return padded_sequences


## Trainer for ProphetNet
class ProphetNetTrainer(Seq2SeqTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, gamma=1, n=2):
        alpha = gamma / n
        
        labels = inputs.get("labels")
        labels_ngram = inputs.get("labels_ngram")
        
        outputs = model(input_ids=inputs.get("input_ids"), 
                        attention_mask=inputs.get("attention_mask"), 
                        decoder_input_ids=inputs.get("decoder_input_ids"), 
                        decoder_attention_mask=inputs.get("decoder_attention_mask"))
        
        logits = outputs.logits
        logits_ngram = outputs.logits_ngram
        
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(logits.view(-1,self.model.config.vocab_size), labels.view(-1))
        ngram_loss = loss_fct(logits_ngram.reshape(-1,self.model.config.vocab_size), labels_ngram.view(-1))
        loss = alpha*lm_loss + alpha*ngram_loss
        
        return (loss, outputs) if return_outputs else loss

## Configure data, tokenizer and model
if model_size == "base":
    pretrain_data = load_dataset("json", data_files="./pretrain_data_base.json")["train"]
    pretrain_data = pretrain_data.train_test_split(test_size=0.1, seed=2022)

    config = XLMProphetNetConfig(vocab_size=32012,
                                 hidden_size=768,
                                 encoder_ffn_dim=3072,
                                 num_encoder_layers=6,
                                 num_encoder_attention_heads=16,
                                 decoder_ffn_dim=3072,
                                 num_decoder_layers=6,
                                 num_decoder_attention_heads=16,
                                 max_position_embeddings=512,
                                 ngram=2)
    model = XLMProphetNetForConditionalGeneration(config)
    tokenizer = XLMProphetNetTokenizer("./ko_tokenizer_base.model", model_max_length=512)
else:
    pretrain_data = load_dataset("json", data_files="./pretrain_data_large.json")["train"]
    pretrain_data = pretrain_data.train_test_split(test_size=0.1, seed=2022)

    config = XLMProphetNetConfig(vocab_size=32012,
                                 hidden_size=1024,
                                 encoder_ffn_dim=4096,
                                 num_encoder_layers=12,
                                 num_encoder_attention_heads=16,
                                 decoder_ffn_dim=4096,
                                 num_decoder_layers=12,
                                 num_decoder_attention_heads=16,
                                 max_position_embeddings=512,
                                 ngram=2)
    model = XLMProphetNetForConditionalGeneration(config)
    tokenizer = XLMProphetNetTokenizer("./ko_tokenizer_large.model", model_max_length=512)


## Pre-train
data_collator = ProphetNetDataCollator(tokenizer=tokenizer, max_length=512)

training_args = Seq2SeqTrainingArguments(
    output_dir="./"+model_size,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=100000,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    max_grad_norm=1.0,
    num_train_epochs=1,
    max_steps=1000000,
    lr_scheduler_type="linear",
    warmup_steps=10000,
    logging_strategy="steps",
    logging_steps=100000,
    save_strategy="steps",
    save_steps=100000,
    remove_unused_columns=False
)

trainer = ProphetNetTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=pretrain_data["train"],
    eval_dataset=pretrain_data["test"],
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model("./"+model_size)



