import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
START_IDX = 0
LT_SUBSTITUTE = '|'
PAD_MULTIPLE = 64
LOWER_LITERALS = False
LITERAL_RE = re.compile(r"'[^']*'")
AUGMENT = ''
CITY_RE = re.compile(r"(city_\d+)\.city_name = '([^']*)'")
COL_LITERAL_RE = re.compile(r"\w+\.(\w+) = '[^']*'")
PLACE_COLUMNS = {'airport_code', 'airport_name', 'state_code', 'state_name', 'country_name'}
ANONYMIZE = False
PLACEHOLDER = 'loc{}'
PLACEHOLDER_RE = re.compile(r"'loc(\d+)'", re.IGNORECASE)


def pad_batch(seqs):
    '''Right-pad a list of 1-D id tensors with PAD_IDX to a common length that is a multiple of PAD_MULTIPLE.'''
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    extra = (-padded.shape[1]) % PAD_MULTIPLE
    return torch.nn.functional.pad(padded, (0, extra), value=PAD_IDX) if extra else padded


def preprocess_sql(sql):
    '''Preprocess a SQL query'''
    sql = sql.replace('<', LT_SUBSTITUTE)
    if LOWER_LITERALS:
        sql = LITERAL_RE.sub(lambda m: m.group(0).lower(), sql)
    return sql


def postprocess_sql(sql):
    '''Undo preprocess_sql'''
    if LOWER_LITERALS:
        sql = LITERAL_RE.sub(lambda m: m.group(0).upper(), sql).replace("='", "= '")
    return sql.replace(LT_SUBSTITUTE, '<')


def city_spans(question, city):
    '''Occurence of a city name in a lowercase question'''
    pattern = r'(?<![\w.])' + re.escape(city.lower()) + r'(?!\w)'
    return [m.span() for m in re.finditer(pattern, question)]


def replace_cities(question, sql, mapping):
    '''Replace cities'''
    spans = sorted(((a, b, new.lower()) for old, new in mapping.items() for a, b in city_spans(question, old)), reverse=True)
    for a, b, new in spans:  # right to left
        question = question[:a] + new + question[b:]
    sql = CITY_RE.sub(lambda m: f"{m.group(1)}.city_name = '{mapping.get(m.group(2), m.group(2))}'", sql)
    return question, sql


def augment_cities(nl, sql, db_path, mode, seed=0):
    '''Return the training pairs plus city-swapped copies'''
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        all_cities = sorted({row[0] for row in conn.execute('SELECT DISTINCT city_name FROM city')})
    rng = random.Random(seed)
    new_nl, new_sql, n_mirror, n_swap = list(nl), list(sql), 0, 0
    for q, s in zip(nl, sql):
        cols = COL_LITERAL_RE.findall(s)
        cities = sorted(set(c for _, c in CITY_RE.findall(s)))
        if not cities or len(cols) != s.count("'") // 2 or PLACE_COLUMNS & set(cols):
            continue
        spans = [city_spans(q, c) for c in cities]
        if any(len(sp) != 1 for sp in spans):
            continue
        flat = sorted(sp[0] for sp in spans)
        if any(b1 > a2 for (_, b1), (a2, _) in zip(flat, flat[1:])):  # overlapping names
            continue
        if len(cities) == 2:
            a, b = cities
            q2, s2 = replace_cities(q, s, {a: b, b: a})
            new_nl.append(q2); new_sql.append(s2); n_mirror += 1
        for _ in range({'mirror+swap': 1, 'mirror+swap2': 2}.get(mode, 0)):
            others = rng.sample([c for c in all_cities if c not in cities], len(cities))
            q2, s2 = replace_cities(q, s, dict(zip(cities, others)))
            new_nl.append(q2); new_sql.append(s2); n_swap += 1
    print(f'augment_cities ({mode}): {len(nl)} pairs + {n_mirror} mirrored + {n_swap} random-city = {len(new_nl)}')
    return new_nl, new_sql


def db_cities(db_path):
    '''City names in the database, longest first'''
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        return sorted({row[0] for row in conn.execute('SELECT DISTINCT city_name FROM city')}, key=len, reverse=True)


def anonymize_cities(question, sql, cities):
    '''Return (question, sql, order): cities named in the question replaced by loc<i>'''
    found = []
    for city in cities:
        for a, b in city_spans(question, city):
            if not any(a < e and s < b for s, e, _ in found):
                found.append((a, b, city))
    order = []
    for _, _, city in sorted(found):
        if city not in order:
            order.append(city)
    index = {city: i for i, city in enumerate(order)}
    for a, b, city in sorted(found, reverse=True):
        question = question[:a] + PLACEHOLDER.format(index[city]) + question[b:]
    if sql is not None:
        sql = CITY_RE.sub(lambda m: f"{m.group(1)}.city_name = '{PLACEHOLDER.format(index[m.group(2)])}'"
                          if m.group(2) in index else m.group(0), sql)
    return question, sql, order


def restore_cities(sql, order):
    '''Undo anonymize_cities on a decoded query'''
    return PLACEHOLDER_RE.sub(lambda m: f"'{order[int(m.group(1))]}'" if int(m.group(1)) < len(order) else m.group(0), sql)


class T5Dataset(Dataset):

    def __init__(self, data_folder, split, prefix=''):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output.
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        # Optional task prefix
        self.prefix = prefix
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # For stats
        self.nl = load_lines(f'{data_folder}/{split}.nl')
        self.sql = None if split == 'test' else load_lines(f'{data_folder}/{split}.sql')
        if split == 'train' and AUGMENT:
            self.nl, self.sql = augment_cities(self.nl, self.sql, f'{data_folder}/flight_database.db', AUGMENT)
        self.city_order = None
        if ANONYMIZE:
            cities = db_cities(f'{data_folder}/flight_database.db')
            out = [anonymize_cities(q, s, cities) for q, s in zip(self.nl, self.sql or [None] * len(self.nl))]
            self.nl = [q for q, _, _ in out]
            self.city_order = [order for _, _, order in out]
            if self.sql is not None:
                self.sql = [s for _, s, _ in out]
        self.encoder_input = [torch.tensor(ids) for ids in tokenizer([self.prefix + q for q in self.nl])["input_ids"]]

        self.decoder_input, self.decoder_target = [], []
        if split == 'test':
            return
        targets = tokenizer([preprocess_sql(q) for q in self.sql])["input_ids"]
        for tgt in targets:
            self.decoder_input.append(torch.tensor([START_IDX] + tgt[:-1]))
            self.decoder_target.append(torch.tensor(tgt))

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        if self.split == 'test':
            return self.encoder_input[idx]
        return self.encoder_input[idx], self.decoder_input[idx], self.decoder_target[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_input, decoder_input, decoder_target = zip(*batch)

    encoder_ids = pad_batch(encoder_input)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_batch(decoder_input)
    decoder_targets = pad_batch(decoder_target)
    initial_decoder_inputs = torch.full((len(batch), 1), START_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_batch(batch)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), START_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, prefix=""):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, prefix)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, prefix=""):
    train_loader = get_dataloader(batch_size, "train", prefix)
    dev_loader = get_dataloader(test_batch_size, "dev", prefix)
    test_loader = get_dataloader(test_batch_size, "test", prefix)

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

# def load_prompting_data(data_folder):
#     # TODO
#     return train_x, train_y, dev_x, dev_y, test_x
