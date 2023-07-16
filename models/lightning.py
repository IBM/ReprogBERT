import os.path

import pytorch_lightning as pl
from transformers import BertForMaskedLM, BertConfig
from models.ReprogBert import BertForMaskedLMProt
from models.ReprogBert import BertConfigProtein
import numpy as np
import torch
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import logging
from tokenizers import Tokenizer
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
log = logger


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ProGen(metaclass=Singleton):
    def __init__(self, progen_dir):
        import sys
        sys.path.insert(0, progen_dir)
        from models.progen.modeling_progen import ProGenForCausalLM
        self.proGenLM = ProGenForCausalLM.from_pretrained(os.path.join(progen_dir, 'checkpoints/progen2-base')).cuda()
        self.proGenTokenizer = Tokenizer.from_str(open(os.path.join(progen_dir, 'tokenizer.json'), 'r').read())

    def proGenLoss(self, input_ids, mask, infill_mask, tokenizer, criterion, split_loss=True):

        device = input_ids.device

        input_ids_clean = []
        infill_mask_clean = []
        num_aa_perseq = []

        for p, tm, im, in zip(input_ids, mask, infill_mask):
            input_ids_clean.append(p[tm > 0][1:-1])  # remove [CLS], [SEP]
            infill_mask_clean.append(im[tm > 0][1:-1])
            num_aa_perseq.append(len(infill_mask_clean[-1]))

        input_ids_clean = pad_sequence(input_ids_clean, batch_first=True)
        infill_mask_clean = pad_sequence(infill_mask_clean, batch_first=True)

        pred_str = tokenizer.batch_decode(input_ids_clean, skip_special_tokens=True)
        pred_str = [''.join(s.split()) for s in pred_str]

        pred_str_hacked = []
        # hack to add missing AA in case sequence was misformed (early in training)
        for s, n in zip(pred_str, num_aa_perseq):
            pred_str_hacked.append(s + 'S'*(n-len(s)))

        enc_tok = self.proGenTokenizer.encode_batch(pred_str_hacked)
        inps = pad_sequence([torch.tensor(e.ids) for e in enc_tok], batch_first=True).to(device)
        attn = pad_sequence([torch.tensor(e.attention_mask) for e in enc_tok], batch_first=True).to(device)

        with torch.no_grad():
            out = self.proGenLM(inps, attention_mask=attn)

        lgts = out.logits
        lgts = lgts[:, :-1]  # shift
        trgt = inps[:, 1:]
        msk = infill_mask_clean[:, 1:]

        if split_loss:
            L = []
            for l,t,m in zip(lgts, trgt, msk):
                L.append(criterion(l[m], t[m]).item())
        else:
            L = criterion(lgts[msk], trgt[msk]).item()

        return L


class ProtBERTLight(pl.LightningModule):

    def __init__(self,
                 model_type,
                 cache,
                 bert_type,
                 tokenizer_name,
                 lr,
                 num_samples,
                 proGen_dir=None):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False, cache_dir=cache)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.lr = lr
        self.num_samples = num_samples
        self.proGen_dir = proGen_dir

        if model_type == 'prot':
            self.model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert", cache_dir=cache)

        elif model_type == 'base':
            model = BertForMaskedLM.from_pretrained(bert_type, cache_dir=cache)
            state_dict = model.state_dict()

            # Remove vocab_size weights
            del state_dict["bert.embeddings.word_embeddings.weight"]
            del state_dict["cls.predictions.decoder.weight"]
            del state_dict["cls.predictions.decoder.bias"]
            del state_dict["cls.predictions.bias"]

            # define new model with custom class classifier
            config = BertConfig.from_pretrained(bert_type, cache_dir=cache)
            config.vocab_size = len(self.tokenizer)
            model = BertForMaskedLM(config)

            model.load_state_dict(state_dict, strict=False)

            self.model = model

            for name, param in model.named_parameters():  # freeze the model network
                if name == 'bert.embeddings.word_embeddings.weight':
                    continue
                param.requires_grad = False

        elif model_type == 'reprog':
            model = BertForMaskedLM.from_pretrained(bert_type, cache_dir=cache)
            state_dict = model.state_dict()

            config = BertConfigProtein.from_pretrained(bert_type, cache_dir=cache)
            config.vocab_size_protein = len(self.tokenizer)
            config.pad_token_id_prot = self.tokenizer.pad_token_id

            model = BertForMaskedLMProt(config)

            model.load_state_dict(state_dict, strict=False)

            for name, param in model.named_parameters():  # freeze the model network
                if name == 'bert.embeddings.theta.weight' or name == 'cls.predictions.gamma.weight':
                    continue
                else:
                    param.requires_grad = False

            self.model = model

    def forward(self, src, src_mask=None, tgt=None, tgt_mask=None):
        output = self.model(input_ids=src, attention_mask=src_mask, labels=tgt, return_dict=True)
        loss = output.loss
        logits = output.logits

        return loss, logits

    def shared_step(self, batch):
        pdbs, src_ids, src_mask, tgt_ids, tgt_mask, infill_loc_indices, infill_mask = batch

        loss, logits = self(src_ids, src_mask, tgt_ids, tgt_mask)

        pred_ids = torch.argmax(logits, dim=-1)

        pred_ids_smpls = None
        pred_ids_bstsmpl = None
        diversity = None
        loss_elem_gpt = 0

        num_elem = infill_mask.sum().item()

        if self.trainer.testing and self.num_samples > 0:

            progen = ProGen(self.proGen_dir)

            smpls_list = []
            diversity_list = []
            best_smpl_all_list = []

            for i, entry in enumerate(torch.nn.functional.softmax(logits, -1)):
                smpls = torch.multinomial(entry, self.num_samples, replacement=True).T

                smpls_tgt = tgt_ids[i].clone().repeat(self.num_samples, 1)
                for j, s in enumerate(smpls):
                    smpls_tgt[j][infill_mask[i]] = s[infill_mask[i]]

                smpls_list.append(smpls_tgt.unsqueeze(0))

                masks = infill_mask[i].clone().repeat(self.num_samples, 1)
                msk = masks.unsqueeze(0) * masks.unsqueeze(1)
                mtch = smpls_tgt.unsqueeze(0) == smpls_tgt.unsqueeze(1)
                mtch_msk = mtch * msk
                div = mtch_msk.sum() / msk.sum()
                diversity_list.append(div.item())

                tgt_mask_i = tgt_mask[i].clone().repeat(self.num_samples, 1)
                infill_mask_i = infill_mask[i].clone().repeat(self.num_samples, 1)
                scores = progen.proGenLoss(smpls, tgt_mask_i, infill_mask_i, self.tokenizer, self.criterion)

                best_smpl = smpls[np.argmin(scores)]
                best_smpl_all = tgt_ids[i].clone()
                best_smpl_all[infill_mask[i]] = best_smpl[infill_mask[i]]
                best_smpl_all_list.append(best_smpl_all.unsqueeze(0))

            pred_ids_smpls = torch.cat(smpls_list, 0)
            pred_ids_bstsmpl = torch.cat(best_smpl_all_list, 0)

            matches = tgt_ids[infill_mask] == pred_ids_bstsmpl[infill_mask]
            succ_matches = matches.sum().item()
            total_matches = len(matches)
            diversity = np.mean(diversity_list)

            L = progen.proGenLoss(pred_ids, tgt_mask, infill_mask, self.tokenizer, self.criterion, split_loss=False)
            loss_elem_gpt = L*num_elem
        else:
            matches = tgt_ids[infill_mask] == pred_ids[infill_mask]
            succ_matches = matches.sum().item()
            total_matches = len(matches)

        L = self.criterion(logits[infill_mask], tgt_ids[infill_mask]).item()
        loss_elem = L * num_elem

        return {'loss': loss,
                'logits': logits,
                'num_elem': num_elem,
                'loss_elem': loss_elem,
                'loss_elem_gpt': loss_elem_gpt,
                'diversity': diversity,
                'succ_matches': succ_matches,
                'num_matches': total_matches,
                'pdbs': pdbs,
                'pred_ids_smpls': pred_ids_smpls,
                'pred_ids_bstsmpl': pred_ids_bstsmpl,
                'pred_ids': pred_ids}

    def training_step(self, batch, batch_idx):
        step_output = self.shared_step(batch)
        loss = step_output['loss']
        self.log_dict({'train_CE': loss}, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        step_output = self.shared_step(batch)
        loss = step_output['loss']
        self.log_dict({'valid_CE': loss,
                       'valid_AAR': 100 * step_output['succ_matches'] / step_output['num_matches']},
                        on_step=True, on_epoch=True, logger=True)

        return {'succ_matches': step_output['succ_matches'],
                'num_matches': step_output['num_matches']}

    def validation_epoch_end(self, outputs) -> None:
        succ, tot = 0, 0
        for out in outputs:
            succ += out['succ_matches']
            tot += out['num_matches']

        self.log_dict({'AAR_val': 100 * succ / tot})

    def test_step(self, batch, batch_idx):

        step_output = self.shared_step(batch)
        pred_ids = step_output['pred_ids']

        pdbs = batch[0]
        src_ids = batch[1]
        tgt_ids = batch[3]

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        src_str = self.tokenizer.batch_decode(src_ids)
        tgt_str = self.tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)

        pred_str_smpls = None
        pred_str_bstsmpl = None
        if self.num_samples > 0:
            pred_ids_smpls = step_output['pred_ids_smpls']
            pred_str_smpls = []
            for smpls in pred_ids_smpls:
                pred_str_smpls.append(self.tokenizer.batch_decode(smpls, skip_special_tokens=True))

            pred_ids_bstsmpl = step_output['pred_ids_bstsmpl']
            pred_str_bstsmpl = self.tokenizer.batch_decode(pred_ids_bstsmpl, skip_special_tokens=True)

        return {'pred_str': pred_str,
                'src_str': src_str,
                'tgt_str': tgt_str,
                'pdbs': pdbs,
                'loss_elem_gpt': step_output['loss_elem_gpt'],
                'num_elem': step_output['num_elem'],
                'div': step_output['diversity'],
                'succ_matches': step_output['succ_matches'],
                'num_matches': step_output['num_matches'],
                'pred_str_smpls': pred_str_smpls,
                'pred_str_bstsmpl': pred_str_bstsmpl}

    def test_epoch_end(self, outputs):

        pred_str_all = []
        src_str_all = []
        tgt_str_all = []
        pdbs_all = []
        div_all = []
        loss_elem_gpt, num_elem = 0, 0
        succ, tot = 0, 0
        succ_all, tot_all = [], []

        for out in outputs:
            pdbs_all += out['pdbs']
            pred_str_all += out['pred_str']
            src_str_all += out['src_str']
            tgt_str_all += out['tgt_str']
            loss_elem_gpt += out['loss_elem_gpt']
            num_elem += out['num_elem']
            succ += out['succ_matches']
            tot += out['num_matches']
            div_all.append(out['div'])
            succ_all.append(out['succ_matches'])
            tot_all.append(out['num_matches'])

        fasta_file = open(f'{self.trainer.default_root_dir}/pred.fasta', 'w')
        for i, pr in enumerate(pred_str_all):
            fasta_file.write(">" + f'{pdbs_all[i]}' + "\n" + "".join(pr.split()) + "\n")
        fasta_file.close()

        fasta_file = open(f'{self.trainer.default_root_dir}/true.fasta', 'w')
        for i, pr in enumerate(tgt_str_all):
            fasta_file.write(">" + f'{pdbs_all[i]}' + "\n" + "".join(pr.split()) + "\n")
        fasta_file.close()

        pred_str_bstsmpl = []
        pred_str_smpls = []

        for out in outputs:
            pred_str_bstsmpl += out['pred_str_bstsmpl']
            pred_str_smpls += out['pred_str_smpls']

        fasta_file = open(f'{self.trainer.default_root_dir}/pred_bstsmpl.fasta', 'w')
        for i, pr in enumerate(pred_str_bstsmpl):
            fasta_file.write(">" + f'{pdbs_all[i]}' + "\n" + "".join(pr.split()) + "\n")
        fasta_file.close()

        fasta_file = open(f'{self.trainer.default_root_dir}/pred_smpls.fasta', 'w')
        for i, pr in enumerate(pred_str_smpls):
            for j in range(self.num_samples):
                preds = pr[j]
                fasta_file.write(">" + f'{pdbs_all[i]}' + "\n" + "".join(preds.split()) + "\n")
        fasta_file.close()

        fasta_file = open(f'{self.trainer.default_root_dir}/masked.fasta', 'w')
        for i, pr in enumerate(src_str_all):
            s = "".join(pr.split())
            s = s.replace('[CLS]', '')
            s = s.replace('[SEP]', '')
            s = s.replace('[PAD]', '')
            fasta_file.write(">" + f'{pdbs_all[i]}' + "\n" + s + "\n")
        fasta_file.close()

        log.info(
            f'PPL_test = {np.exp(loss_elem_gpt / num_elem)}, '
            f'AAR_test = {100 * succ / tot}, '
            f'DIV_test = {100 - 100 * np.mean(div_all)}')

        self.log_dict({'PPL_test': np.exp(loss_elem_gpt / num_elem),
                       'AAR_test': 100 * succ / tot,
                       'DIV_test': 100 - 100 * np.mean(div_all)})

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer
