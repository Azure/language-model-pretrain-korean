import sentencepiece as spm




## Specify which tokenizer to train (base/large)
tokenizer_size = "base"


## Train
if tokenizer_size == "base":
    spm.SentencePieceTrainer.train(input="./ko_corpus_base.txt", 
                                   model_prefix="ko_tokenizer_base", 
                                   vocab_size=32000, 
                                   pad_id=3,
                                   unk_piece="[UNK]",
                                   pad_piece="[PAD]",
                                   user_defined_symbols=["[SEP]", "[CLS]", "[MASK]", "[X_SEP]"],
                                   input_sentence_size=0,
                                   shuffle_input_sentence=True,
                                   train_extremely_large_corpus=True)
else:
    spm.SentencePieceTrainer.train(input="./ko_corpus_large.txt", 
                                   model_prefix="ko_tokenizer_large", 
                                   vocab_size=32000, 
                                   pad_id=3,
                                   unk_piece="[UNK]",
                                   pad_piece="[PAD]",
                                   user_defined_symbols=["[SEP]", "[CLS]", "[MASK]", "[X_SEP]"],
                                   input_sentence_size=0,
                                   shuffle_input_sentence=True,
                                   train_extremely_large_corpus=True)



