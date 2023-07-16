import logging
import pytorch_lightning as pl
from data.dataset import create_dataloader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
import os
from models.lightning import ProtBERTLight
import argparse
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
log = logger


def main(args):

    # pl.seed_everything(42)

    # load all dataloaders
    dataloaders = create_dataloader(args.dataset,
                                    args.tokenizer_name,
                                    args.cache,
                                    args.bsize,
                                    args.bsize_eval,
                                    args.num_data_workers)

    # set tb logger
    tb_dir = os.path.join(args.root_dir, args.exp_dir, "tb_logs")
    tb_logger = TensorBoardLogger(tb_dir, version=0)

    checkpoint_dir = os.path.join(args.root_dir, args.exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(args.checkpoint):
        restart_model = args.checkpoint
    else:
        if args.resume:
            file = os.path.join(checkpoint_dir, 'last.ckpt')
            if os.path.exists(file):
                restart_model = file
            else:
                log.info(f'The model {file} does not exist' )
                restart_model = None
        else:
            restart_model = None

    if args.run == 'train':
        model = ProtBERTLight(model_type=args.model_type,
                              cache=args.cache,
                              bert_type=args.bert_type,
                              tokenizer_name=args.tokenizer_name,
                              lr=args.lr,
                              num_samples=args.num_samples)

        checkpoint_callback = ModelCheckpoint(
            monitor='valid_AAR_epoch',
            dirpath=checkpoint_dir,
            filename='{epoch:02d}-{valid_AAR_epoch:.2f}',
            save_top_k=3,
            mode='max',
            save_last=True
        )

        trainer = pl.Trainer.from_argparse_args(args,
                                                default_root_dir=os.path.join(args.root_dir, args.exp_dir),
                                                logger=tb_logger,
                                                callbacks=[checkpoint_callback, RichProgressBar(args.bar)])

        trainer.fit(model=model,
                    train_dataloaders=dataloaders['train'],
                    val_dataloaders=dataloaders['validation'],
                    ckpt_path=restart_model)

    elif args.run == 'test':
        if restart_model is None:
            log.info(f'The model checkpoint was not found, cannot do testing')
            return

        model = ProtBERTLight.load_from_checkpoint(restart_model,
                                                   strict=False,
                                                   proGen_dir=args.progen_dir
                                                   )
        trainer = pl.Trainer.from_argparse_args(args,
                                                default_root_dir=os.path.join(args.root_dir, args.exp_dir),
                                                logger=tb_logger)
        trainer.test(model, dataloaders=dataloaders['test'])

    elif args.run == 'inference':
        if restart_model is None:
            log.info(f'The model checkpoint was not found, cannot do inference')
            return

        model = ProtBERTLight.load_from_checkpoint(restart_model, strict=False)

        spaced_seq = ' '.join(list(args.single_input))

        # prepare input
        batch = model.tokenizer(
            spaced_seq,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt',
            verbose=True
        )

        src_ids = batch["input_ids"]
        mask = [False]*len(src_ids[0])  # Initiate mask with False values

        for i in range(len(src_ids[0])):
            if src_ids[0][i] == model.tokenizer.unk_token_id:
                src_ids[0][i] = model.tokenizer.mask_token_id
                mask[i] = True

        _, logits = model(src_ids)

        smpls = torch.multinomial(torch.nn.functional.softmax(logits[0], -1), args.num_samples, replacement=True).T

        smpls_adj = src_ids.clone().repeat(args.num_samples, 1)
        for i, s in enumerate(smpls):
            smpls_adj[i][mask] = s[mask]

        pred_str_smpls = model.tokenizer.batch_decode(smpls_adj, skip_special_tokens=True)

        fasta_file = open(os.path.join(args.root_dir, args.exp_dir, 'inference_smpl.fasta'),'w')
        for i, pr in enumerate(pred_str_smpls):
            fasta_file.write(">" + f'sample_{i}' + "\n" + "".join(pr.split()) + "\n")
        fasta_file.close()

        log.info(f'Inference completed')
    else:
        raise ValueError("Invalid run mode. Allowed modes are 'train', 'test', and 'inference'.")


if __name__ == "__main__":

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Arguments')

    parser.add_argument("--run", type=str, default='train')
    parser.add_argument("--loging_level", choices=["debug", "info"], default="info", help="logging level")
    parser.add_argument("--single_input", type=str, default='QVQLVESGGGFAQAGGSLRLSCAAS********MGWFRQAPGKEREFVAGISWSGSTKYTDSVKGRFTISRDNAKNTVHLQMNNLTPEDTAVYYCAQSRAIEADDSRGYDYWGQGTQVT')
    parser.add_argument('--root_dir', type=str, default='output')
    parser.add_argument('--exp_dir', type=str, default='base_sabdab1')
    parser.add_argument('--model_type', type=str, default='reprog')
    parser.add_argument('--bert_type', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer_name', type=str, default='Rostlab/prot_bert')
    parser.add_argument('--progen_dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='sabdab3')
    parser.add_argument('--bsize', type=int, default=32)
    parser.add_argument('--bsize_eval', type=int, default=32)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--cache', type=str, default='cache')
    parser.add_argument('--num_data_workers', type=int, default=4)
    parser.add_argument('--bar', default=0, type=int)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
