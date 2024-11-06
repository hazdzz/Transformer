import argparse
import gc
import os
import random
import sacrebleu
import warnings
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from model.transformer import Transformer
from utils import early_stopping, opt, metrics, mt_data_prep
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_env(seed = 3407) -> None:
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_parameters():
    parser = argparse.ArgumentParser(description='Transformer for wmt-14 data')
    parser.add_argument('--config', type=str, default='mt_config.yaml', help='Path to the yaml configuration file')
    parser.add_argument('--task', type=str, default='en-de', choices=['de-en', 'fr-en', 'en-de', 'en-fr'], help='Name of the task')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    task_config = config[args.task]

    for key, value in task_config.items():
        key_type = type(value)
        if key_type is bool:
            action = 'store_false' if value else 'store_true'
            parser.add_argument(f'--{key}', action=action, default=value, help=f'{key} (default: {value})')
        elif key_type in [int, float, str]:
            parser.add_argument(f'--{key}', type=key_type, default=value, help=f'{key} (default: {value})')
        else:
            raise ValueError(f"Unsupported type for key: {key}")

    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        device = torch.device('cpu')
        gc.collect()

    return args, device


def prepare_data(args):
    if args.dataset == 'de-en':
        src_lang = 'de'
        tgt_lang = 'en'
    elif args.dataset == 'fr-en':
        src_lang = 'fr'
        tgt_lang = 'en'
    elif args.dataset == 'en-de':
        src_lang = 'en'
        tgt_lang = 'de'
    elif args.dataset == 'en-fr':
        src_lang = 'en'
        tgt_lang = 'fr'

    train_dataloader, valid_dataloader, test_dataloader, \
        src_lang_transform, tgt_lang_transform, \
        src_tokenizer, tgt_tokenizer, \
        src_vocab_size, tgt_vocab_size, \
        src_pad_idx, tgt_pad_idx, tgt_sos_idx, \
        special_symbols = mt_data_prep.get_data(src_lang, tgt_lang, args.batch_size, args.num_workers)

    return train_dataloader, valid_dataloader, test_dataloader, \
        src_lang_transform, tgt_lang_transform, \
        src_tokenizer, tgt_tokenizer, \
        src_vocab_size, tgt_vocab_size, \
        src_pad_idx, tgt_pad_idx, tgt_sos_idx, \
        special_symbols


def prepare_model(args, src_pad_idx, tgt_pad_idx, tgt_sos_idx, src_vocab_size, tgt_vocab_size, special_symbols, device):
    torch.autograd.set_detect_anomaly(True)

    model = Transformer(args, src_pad_idx, tgt_pad_idx, tgt_sos_idx, src_vocab_size, tgt_vocab_size).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=special_symbols["<pad>"])

    es = early_stopping.EarlyStopping(delta=0.0, 
                                      patience=args.patience,
                                      verbose=True, 
                                      path="transformer_" + args.dataset + ".pt")
    
    if args.optimizer == 'adamw': # default
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'nadamw':
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, decoupled_weight_decay=True)
    elif args.optimizer == 'ademamix':
        optimizer = opt.AdEMAMix(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'ERROR: The {args.optimizer} optimizer is undefined.')
    
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=3, eta_min=0.0005)

    return model, loss_fn, optimizer, scheduler, es


def train(model, dataloader, loss_fn, optim, scheduler, device):
    model.train()

    loss_meter = metrics.AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for src, tgt in dataloader:
        src = src.transpose(0, 1).to(device)
        tgt = tgt.transpose(0, 1).to(device)

        tgt_input = tgt[:, :-1]

        logits = model(src, tgt_input)

        optim.zero_grad()

        tgt_out = tgt[:, 1:]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optim.step()

        loss_meter.update(loss.item(), tgt_out.size(0))

        pbar.set_postfix(loss=loss_meter.avg)
        pbar.update(1)

    # scheduler.step()

    return loss_meter.avg


@torch.no_grad()
def validate(model, dataloader, loss_fn, src_tokenizer, tgt_tokenizer, device):
    model.eval()

    loss_meter = metrics.AverageMeter()
    references = []
    hypotheses = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validating")

    for src, tgt in dataloader:
        src = src.transpose(0, 1).to(device)
        tgt = tgt.transpose(0, 1).to(device)

        tgt_input = tgt[:, :-1]

        logits = model(src, tgt_input)

        tgt_out = tgt[:, 1:]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss_meter.update(loss.item(), tgt_out.size(0))

        predicted_tokens = logits.argmax(-1)
        predicted_sentences = [tgt_tokenizer.decode(predicted.tolist(), skip_special_tokens=True) for predicted in predicted_tokens.transpose(0, 1)]
        target_sentences = [tgt_tokenizer.decode(tgt.tolist(), skip_special_tokens=True) for tgt in tgt_out.transpose(0, 1)]

        hypotheses.extend(predicted_sentences)
        references.extend([[tgt_sentence] for tgt_sentence in target_sentences])

        pbar.set_postfix(loss=loss_meter.avg)
        pbar.update(1)

    bleu = sacrebleu.corpus_bleu(hypotheses, references)

    return bleu.score, loss_meter.avg


@torch.no_grad()
def test(args, model, dataloader, loss_fn, src_tokenizer, tgt_tokenizer, device):
    model.load_state_dict(torch.load("transformer_" + args.dataset + ".pt"))
    model.eval()

    loss_meter = metrics.AverageMeter()
    references = []
    hypotheses = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")

    for src, tgt in dataloader:
        src = src.transpose(0, 1).to(device)
        tgt = tgt.transpose(0, 1).to(device)

        tgt_input = tgt[:, :-1]

        logits = model(src, tgt_input)

        tgt_out = tgt[:, 1:]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss_meter.update(loss.item(), tgt_out.size(0))

        predicted_tokens = logits.argmax(-1)
        predicted_sentences = [tgt_tokenizer.decode(predicted.tolist(), skip_special_tokens=True) for predicted in predicted_tokens.transpose(0, 1)]
        target_sentences = [tgt_tokenizer.decode(tgt.tolist(), skip_special_tokens=True) for tgt in tgt_out.transpose(0, 1)]

        hypotheses.extend(predicted_sentences)
        references.extend([[tgt_sentence] for tgt_sentence in target_sentences])

        pbar.set_postfix(loss=loss_meter.avg)
        pbar.update(1)

    bleu = sacrebleu.corpus_bleu(hypotheses, references)

    return bleu.score, loss_meter.avg


def run(args, model, train_dataloader, valid_dataloader, loss_ce, optimizer, scheduler, src_tokenizer, tgt_tokenizer, es, device):
    for _ in range(1, args.epochs + 1):
        loss_train = train(model, train_dataloader, loss_ce, optimizer, scheduler, device)
        bleu_val, loss_val = validate(model, valid_dataloader, loss_ce, src_tokenizer, tgt_tokenizer, device)
        print(f'train loss: {loss_train: .2f}')
        print(f'val bleu: {bleu_val: .2f}')
        print(f'val loss: {loss_val: .2f}')

        es(loss_val, model)
        if es.early_stop:
            print("Early stopping")
            break

    return loss_train, bleu_val, loss_val


if __name__ == '__main__':
    SEED = 3704
    set_env(SEED)

    warnings.filterwarnings("ignore", category=UserWarning)

    args, device = get_parameters()
    train_dataloader, valid_dataloader, test_dataloader, \
        src_lang_transform, tgt_lang_transform, \
        src_tokenizer, tgt_tokenizer, \
        src_vocab_size, tgt_vocab_size, \
        src_pad_idx, tgt_pad_idx, tgt_sos_idx, \
        special_symbols = prepare_data(args)
    
    model, loss_fn, optimizer, scheduler, es = prepare_model(args, src_pad_idx, tgt_pad_idx, tgt_sos_idx, src_vocab_size, tgt_vocab_size, special_symbols, device)
    loss_train, bleu_val, loss_val = run(args, model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, src_tokenizer, tgt_tokenizer, es, device)
    bleu_test, loss_test = test(args, model, test_dataloader, loss_fn, src_tokenizer, tgt_tokenizer, device)

    print(f'test bleu: {bleu_test: .2f}')
    print(f'test loss: {loss_test: .2f}')