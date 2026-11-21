import os
import argparse
import re
import sqlite3
import time
import warnings
from tqdm import tqdm

# Must import before torch
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.6')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.5')

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5ForConditionalGeneration
import load_data
from concurrent.futures import ThreadPoolExecutor
from utils import compute_metrics, save_queries_and_records, read_queries, DB_PATH

DEVICE = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('mps') if torch.backends.mps.is_available()
          else torch.device('cpu'))
PAD_IDX = 0
UNK_IDX = 2
EXEC_TIME_LIMIT = 10

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")
    parser.add_argument('--gen_every', type=int, default=1,
                        help="Generate and score dev queries every this many epochs and after the last one; "
                             "the other epochs log only the dev loss. 'best' is chosen among the scored epochs")
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help="Resume a stopped run: how many epochs it finished. Loads the 'last' checkpoint of "
                             "--experiment_name, moves the LR schedule to that point and trains the remaining "
                             "epochs. Pass the same schedule args as the stopped run. AdamW's moment estimates "
                             "aren't saved, so they restart from zero.")
    parser.add_argument('--resume_best_f1', type=float, default=-1,
                        help="Best dev Record F1 of the stopped run, so a worse epoch can't overwrite 'best'")
    parser.add_argument('--init_from', type=str, default='',
                        help="Start training from the weights in this checkpoint folder (e.g. another "
                             "scratch run's best/) instead of a new model; '' = off")
    parser.add_argument('--decoder_input_dropout', type=float, default=0,
                        help="Training only: replace this fraction of the decoder's input tokens with <unk>, so it "
                             "can't predict the SQL from its own history alone; 0 = off")
    parser.add_argument('--max_grad_norm', type=float, default=0,
                        help="Clip the global gradient norm to this value before each step; 0 = no clipping")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--task_prefix', type=str, default='',
                        help="Text put in front of every question, e.g. 'translate English to SQL: '")
    parser.add_argument('--augment', type=str, default='', choices=['', 'mirror', 'mirror+swap', 'mirror+swap2'],
                        help="Add city-swapped copies of the training pairs (see load_data.AUGMENT); '' = off")
    parser.add_argument('--anonymize_cities', action='store_true',
                        help="Replace the cities named in each question with loc0, loc1, ... in the question "
                             "and the SQL, and put the names back after decoding (see load_data.ANONYMIZE)")
    parser.add_argument('--lower_literals', action='store_true',
                        help="Train on SQL with its quoted constants lowercased (as in the questions) and "
                             "uppercase them again after decoding")

    # Generation hyperparameters (used by eval_epoch and test_inference)
    parser.add_argument('--num_beams', type=int, default=1,
                        help="1 = greedy decoding; >1 = beam search with that many beams")
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help="Generation length limit; the longest target is 511 tokens with </s>")
    parser.add_argument('--repair_sql', action='store_true',
                        help="Skip training and decoding: repair the saved greedy dev/test outputs of "
                             "--experiment_name (queries SQLite rejects get balanced brackets if that "
                             "makes them run) and save them as <name>_repair")
    parser.add_argument('--eval_only', action='store_true',
                        help="Skip training; evaluate the saved best checkpoint of --experiment_name "
                             "(e.g. with a different --num_beams)")

    # Data stats arg
    parser.add_argument('--data_stat', action='store_true', help="Compute data statistics for train and dev")

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = args.resume_best_f1  # -1 for a fresh run
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    for epoch in range(args.resume_epoch, args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        if (epoch + 1) % args.gen_every != 0 and epoch != args.max_n_epochs - 1:
            eval_loss = dev_loss(model, dev_loader)
            print(f"Epoch {epoch}: Dev loss: {eval_loss} (no generation this epoch)")
            if args.use_wandb:
                wandb.log({'train/loss': tr_loss, 'dev/loss': eval_loss,
                           'train/lr': optimizer.param_groups[0]['lr']}, step=epoch)
            save_model(checkpoint_dir, model, best=False)
            continue

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
                'train/lr' : optimizer.param_groups[0]['lr'],
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            if args.use_wandb:
                wandb.run.summary['best_dev_record_f1'] = best_f1
                wandb.run.summary['best_epoch'] = epoch
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)
        if args.decoder_input_dropout > 0:
            drop = (torch.rand(decoder_input.shape, device=DEVICE) < args.decoder_input_dropout) & (decoder_input != PAD_IDX)
            decoder_input = decoder_input.masked_fill(drop, UNK_IDX)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        # Clip exploding gradients
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        del logits, loss
        free_mps_cache()

    return total_loss / total_tokens

def free_mps_cache():
    '''On Apple GPU, hand cached memory back after each step'''
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()

def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    loss = dev_loss(model, dev_loader)
    sql_queries = generate_queries(args, model, dev_loader)
    save_queries_and_records(sql_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(gt_sql_pth, model_sql_path, gt_record_path, model_record_path)

    error_rate = sum(msg != '' for msg in error_msgs) / len(error_msgs)
    return loss, record_f1, record_em, sql_em, error_rate

def dev_loss(model, dev_loader):
    '''Cross-entropy on the non-pad dev targets'''
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            del logits, loss
            free_mps_cache()
    return total_loss / total_tokens

def generate_queries(args, model, loader):
    '''
    Generate one SQL query per example of a (dev or test) loader, in order
    '''
    tokenizer = loader.dataset.tokenizer
    gen_device = torch.device('cpu') if DEVICE.type == 'mps' else DEVICE
    model.to(gen_device)
    free_mps_cache()
    sql_queries = []
    with torch.no_grad():
        for batch in tqdm(loader):
            encoder_input, encoder_mask, initial_decoder_inputs = batch[0], batch[1], batch[-1]
            sql_queries.extend(generate_sql(args, model, tokenizer, encoder_input.to(gen_device), encoder_mask.to(gen_device), initial_decoder_inputs.to(gen_device)))
    model.to(DEVICE)
    city_order = loader.dataset.city_order
    if city_order is not None:
        sql_queries = [load_data.restore_cities(q, order) for q, order in zip(sql_queries, city_order)]
    return sql_queries

def sql_status(query):
    """'ok' if SQLite runs the query within EXEC_TIME_LIMIT seconds, 'slow' if it is still running
    then, 'error' if SQLite rejects it."""
    conn = sqlite3.connect(DB_PATH)
    deadline = time.monotonic() + EXEC_TIME_LIMIT
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10000)
    try:
        conn.execute(query).fetchall()
        return 'ok'
    except sqlite3.OperationalError as e:
        return 'slow' if 'interrupted' in str(e) else 'error'
    except Exception:
        return 'error'
    finally:
        conn.close()

def balance_parens(query):
    '''Drop each ')' that closes nothing, then close every '(' still open at the end.'''
    out, depth = [], 0
    for ch in query:
        if ch == ')' and depth == 0:
            continue
        depth += (ch == '(') - (ch == ')')
        out.append(ch)
    return re.sub(r'\s+', ' ', ''.join(out)).strip() + ' )' * depth

def repair_queries(sql_queries):
    '''
    Execution-guided repair
    '''
    with ThreadPoolExecutor(10) as pool:
        status = list(pool.map(sql_status, sql_queries))
    failing = [i for i, st in enumerate(status) if st == 'error']
    candidates = [balance_parens(sql_queries[i]) for i in failing]
    with ThreadPoolExecutor(10) as pool:
        cand_ok = [st == 'ok' for st in pool.map(sql_status, candidates)]
    repaired = list(sql_queries)
    for i, cand, c_ok in zip(failing, candidates, cand_ok):
        if c_ok:
            repaired[i] = cand
    print(f'Repair: {sum(cand_ok)} of {len(failing)} failing queries run after balancing brackets')
    return repaired

def run_repair(args):
    '''repair the dev/test outputs already saved'''
    model_type = 'ft' if args.finetune else 'scr'
    for split in ['dev', 'test']:
        queries = repair_queries(read_queries(f'results/t5_{model_type}_{args.experiment_name}_{split}.sql'))
        name = f't5_{model_type}_{args.experiment_name}_repair_{split}'
        save_queries_and_records(queries, f'results/{name}.sql', f'records/{name}.pkl')
        if split == 'dev':
            sql_em, record_em, record_f1, error_msgs = compute_metrics(
                'data/dev.sql', f'results/{name}.sql', 'records/ground_truth_dev.pkl', f'records/{name}.pkl')
            error_rate = sum(msg != '' for msg in error_msgs) / len(error_msgs)
            print(f"Dev after repair: Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}, "
                  f"{error_rate*100:.2f}% SQL errors")

def generate_sql(args, model, tokenizer, encoder_input, encoder_mask, initial_decoder_inputs):
    '''
    Generate SQL for one batch and turn the token ids back into query strings. Decoding starts
    from initial_decoder_inputs, the same start token the decoder was trained with.
    '''
    generated = model.generate(
        input_ids=encoder_input,
        attention_mask=encoder_mask,
        decoder_input_ids=initial_decoder_inputs,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [load_data.postprocess_sql(t).strip() for t in texts]

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    sql_queries = generate_queries(args, model, test_loader)
    save_queries_and_records(sql_queries, model_sql_path, model_record_path)

def print_data_stats(loaders):
    def stats(nl_ids, sql_ids, unk_id):
        nl_lens = [len(x) for x in nl_ids]
        sql_lens = [len(x) for x in sql_ids]
        return {
            'Number of examples': len(nl_ids),
            'Mean sentence length': sum(nl_lens) / len(nl_lens),
            'Mean SQL query length': sum(sql_lens) / len(sql_lens),
            'Vocabulary size (natural language)': len({t for x in nl_ids for t in x}),
            'Vocabulary size (SQL)': len({t for x in sql_ids for t in x}),
            'Max SQL query length': max(sql_lens),
            'SQL queries containing <unk>': sum(unk_id in x for x in sql_ids),
        }

    tables = {'Table 1: before pre-processing': {}, 'Table 2: after pre-processing': {}}
    for split, loader in loaders.items():
        ds = loader.dataset
        tokenizer = ds.tokenizer
        raw_nl = tokenizer(ds.nl, add_special_tokens=False)['input_ids']
        raw_sql = tokenizer(ds.sql, add_special_tokens=False)['input_ids']
        tables['Table 1: before pre-processing'][split] = stats(
            nl_ids=raw_nl,
            sql_ids=raw_sql,
            unk_id=tokenizer.unk_token_id
        )

        tables['Table 2: after pre-processing'][split] = stats(
            nl_ids=[x.tolist() for x in ds.encoder_input], 
            sql_ids=[x.tolist() for x in ds.decoder_target], 
            unk_id=tokenizer.unk_token_id)

    splits = list(loaders)
    for title, cols in tables.items():
        print(f'\n{title}')
        print(f'  {"Statistic":<36}' + ''.join(f'{s:>10}' for s in splits))
        for key in cols[splits[0]]:
            if key == 'Number of examples' and title.startswith('Table 2'):
                continue
            row = [cols[s][key] for s in splits]
            print(f'  {key:<36}' + ''.join(f'{v:>10.2f}' if isinstance(v, float) else f'{v:>10,}' for v in row))

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb and not args.data_stat:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    load_data.LOWER_LITERALS = args.lower_literals
    load_data.AUGMENT = args.augment
    load_data.ANONYMIZE = args.anonymize_cities
    train_loader, dev_loader, test_loader = load_data.load_t5_data(args.batch_size, args.test_batch_size, args.task_prefix)
    ## Data Statistics
    if args.data_stat:
        print_data_stats({'train': train_loader, 'dev': dev_loader})
        return
    if args.repair_sql:
        run_repair(args)
        return

    if not args.eval_only:
        if args.resume_epoch > 0:
            model = load_model_from_checkpoint(args, best=False)  # where the stopped run left off
        elif args.init_from:
            # Continue from incomplete run
            model = T5ForConditionalGeneration.from_pretrained(args.init_from).to(DEVICE)
        else:
            model = initialize_model(args)
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        if scheduler is not None and args.resume_epoch > 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for _ in range(args.resume_epoch * len(train_loader)):
                    scheduler.step()

        # Train
        train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    # Dev set
    experiment_name = args.experiment_name + (f'_beam{args.num_beams}' if args.num_beams > 1 else '')
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

    if args.use_wandb:
        wandb.run.summary.update({
            'final_dev/loss': dev_loss,
            'final_dev/record_f1': dev_record_f1,
            'final_dev/record_em': dev_record_em,
            'final_dev/sql_em': dev_sql_em,
            'final_dev/error_rate': dev_error_rate,
        })
        wandb.finish()

if __name__ == "__main__":
    main()
