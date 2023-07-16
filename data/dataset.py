import logging
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

logger = logging.getLogger(__name__)
log = logger


def format_sequence(string, length):
    spaced = ' '.join(string[i:i+length] for i in range(0,len(string),length))
    return spaced

def helper_fn_infilling(src_ids, cdr):
    infill_loc_indices = []
    infill_mask = torch.zeros_like(src_ids).bool()
    for i, cdr_batch in enumerate(cdr):
        loc_list = []
        for j, charac in enumerate(cdr_batch):
            if str(charac) == "T":
                loc_list.append(j + 1)
                infill_mask[i,j+1] = True
        infill_loc_indices.append(loc_list)

    max_len = max([len(ele) for ele in infill_loc_indices])

    for idx in range(len(infill_loc_indices)):
        ele = infill_loc_indices[idx]
        ele = ele + [-1 for _ in range(max_len - len(ele))]
        infill_loc_indices[idx] = torch.LongTensor(ele)

    return torch.stack(infill_loc_indices), infill_mask


def batch_infilling_collate(batch, tokenizer):

    key, sequence, cdr = [], [], []

    for sample in batch:
        key.append(sample["Key"])
        seq = format_sequence(sample["Sequence"][:510], 1)
        sequence.append(seq)
        cdr.append(sample["CDR"])

    prepared_batch = tokenizer(
        sequence,
        add_special_tokens=True,
        padding=True,
        return_tensors='pt',
        verbose=True
    )

    src_ids = prepared_batch["input_ids"]
    src_mask_padding = prepared_batch["attention_mask"]
    tgt_ids = src_ids.clone()
    tgt_mask_padding = src_mask_padding.clone()
    infill_loc_indices, infill_mask = helper_fn_infilling(src_ids, cdr)

    for i in range(len(src_ids)):
        ids = src_ids[i]
        for j in infill_loc_indices[i]:
            if j == -1:
                continue
            ids[j] = tokenizer.mask_token_id
        src_ids[i] = ids

    return key, src_ids, src_mask_padding, tgt_ids, tgt_mask_padding, infill_loc_indices, infill_mask


def create_dataset(dataset):
    if dataset == "sabdab1":
        dataset_tag = 'data/sabdab1_script.py'

    elif dataset == "sabdab2":
        dataset_tag = 'data/sabdab2_script.py'

    elif dataset == "sabdab3":
        dataset_tag = 'data/sabdab3_script.py'

    elif dataset == "sabdab_all":
        dataset_tag = 'data/sabdab_all_script.py'

    else:
        raise Exception(f"dataset for {dataset} not defined.")

    ds = load_dataset(dataset_tag)

    train_ds = ds['train']
    dev_ds = ds['validation']
    test_ds = ds['test']

    dataset = {'train': train_ds,
               'validation': dev_ds,
               'test': test_ds}
    
    return dataset


def create_dataloader(dataset,
                      tokenizer_name,
                      cache,
                      bsize,
                      bsize_eval,
                      num_workers):

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False, cache_dir=cache)
    ds = create_dataset(dataset)
    collator = batch_infilling_collate

    train_dl = DataLoader(ds['train'],
                          batch_size=bsize,
                          shuffle=True,
                          collate_fn=lambda batch: collator(batch, tokenizer),
                          num_workers=num_workers,
                          drop_last=True)

    dev_dl = DataLoader(ds['validation'],
                        batch_size=bsize_eval,
                        shuffle=False,
                        collate_fn=lambda batch: collator(batch, tokenizer),
                        num_workers=num_workers,
                        drop_last=False)

    test_dl = DataLoader(ds['test'],
                         batch_size=bsize_eval,
                         shuffle=False,
                         collate_fn=lambda batch: collator(batch, tokenizer),
                         num_workers=num_workers,
                         drop_last=False)

    log.info(f"# train: dataloader: [{len(train_dl)}] batches of  [{len(ds['train'])}] samples)")
    log.info(f"# validation: dataloader: [{len(dev_dl)}]   batches of  [{len(ds['validation'])}] samples)")
    log.info(f"# test : dataloader: [{len(test_dl)}]  batches of  [{len(ds['test'])}] samples)")
    log.info(f"# tokenizer: {tokenizer}")

    return {"train": train_dl,
            "validation": dev_dl,
            "test": test_dl,
            "tokenizer": tokenizer}
