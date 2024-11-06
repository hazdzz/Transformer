import torch

from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset


def get_data(src_lang, tgt_lang, batch_size, num_works):

    special_symbols = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3
    }

    if src_lang == 'de' and tgt_lang == 'en':
        src_lang_model = "bert-base-german-cased"
        tgt_lang_model = "bert-base-uncased"
        dataset = load_dataset("iwslt2017", "iwslt2017-de-en")
    elif src_lang == 'fr' and tgt_lang == 'en':
        src_lang_model = "camembert-base"
        tgt_lang_model = "bert-base-uncased"
        dataset = load_dataset("iwslt2017", "iwslt2017-fr-en")
    elif src_lang == 'en' and tgt_lang == 'de':
        src_lang_model = "bert-base-uncased"
        tgt_lang_model = "bert-base-german-cased"
        dataset = load_dataset("iwslt2017", "iwslt2017-en-de")
    elif src_lang == 'en' and tgt_lang == 'fr':
        src_lang_model = "bert-base-uncased"
        tgt_lang_model = "camembert-base"
        dataset = load_dataset("iwslt2017", "iwslt2017-en-fr")

    # Load tokenizers from transformers library
    src_tokenizer = AutoTokenizer.from_pretrained(src_lang_model)
    tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_lang_model)

    src_vocab_size = len(src_tokenizer) + len(special_symbols)
    tgt_vocab_size = len(tgt_tokenizer) + len(special_symbols)

    def _seq_transform(tokenizer):
        def func(txt_input):
            token_ids = tokenizer(txt_input, add_special_tokens=False, return_tensors=None)['input_ids']
            return torch.cat(
                (torch.tensor([special_symbols["<bos>"]]),
                 torch.tensor(token_ids),
                 torch.tensor([special_symbols["<eos>"]]))
            )
        return func

    src_lang_transform = _seq_transform(src_tokenizer)
    tgt_lang_transform = _seq_transform(tgt_tokenizer)

    src_pad_idx = special_symbols["<pad>"]
    tgt_pad_idx = special_symbols["<pad>"]
    tgt_sos_idx = special_symbols["<bos>"]


    def _collate_fn(batch):
        src_batch, tgt_batch = [], []
        for sample in batch:
            src_sample, tgt_sample = sample["translation"][src_lang], sample["translation"][tgt_lang]
            src_batch.append(src_lang_transform(src_sample.rstrip("\n")))
            tgt_batch.append(tgt_lang_transform(tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=special_symbols["<pad>"])
        tgt_batch = pad_sequence(tgt_batch, padding_value=special_symbols["<pad>"])

        return src_batch, tgt_batch
    

    train_dataloader = DataLoader(dataset["train"],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_works,
                                  collate_fn=_collate_fn,
                                  drop_last=True
                                  )
    valid_dataloader = DataLoader(dataset["validation"],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_works,
                                  collate_fn=_collate_fn,
                                  drop_last=True
                                  )
    test_dataloader = DataLoader(dataset["test"],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_works,
                                 collate_fn=_collate_fn,
                                 drop_last=True
                                 )

    return train_dataloader, valid_dataloader, test_dataloader, \
        src_lang_transform, tgt_lang_transform, \
        src_tokenizer, tgt_tokenizer, \
        src_vocab_size, tgt_vocab_size, \
        src_pad_idx, tgt_pad_idx, tgt_sos_idx, \
        special_symbols


if __name__=="__main__":
    src_lang = 'de'
    tgt_lang = 'en'
    batch_size = 64
    num_works = 2
    train_dataloader, valid_dataloader, test_dataloader, \
        src_lang_transform, tgt_lang_transform, \
        src_tokenizer, tgt_tokenizer, \
        src_vocab_size, tgt_vocab_size, \
        src_pad_idx, tgt_pad_idx, tgt_sos_idx, \
        special_symbols = get_data(src_lang, tgt_lang, batch_size, num_works)
    
    print(test_dataloader.batch_size)