import pandas as pd
import datasets
from os.path import join
import json

_DESCRIPTION = ""
_CITATION = ""
_DATA_ROOT = "/dccstor/imelnyk1/TIGS_AR/data/sabdab/hcdr2_cluster"


class protCDRScript(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "Key": datasets.Value("string"),
                    "Sequence": datasets.Value("string"),
                    "CDR": datasets.Value("string")
                }
            ),
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = join(_DATA_ROOT, "train_data.jsonl")
        valid_path = join(_DATA_ROOT, "val_data.jsonl")
        test_path = join(_DATA_ROOT, "test_data.jsonl")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": valid_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path})
        ]

    def _generate_examples(self, filepath=None): 
        """Generate examples."""
        with open(filepath, 'r') as f:
            data = f.readlines()

        for id_, ex in enumerate(data):
            entry = json.loads(ex)
            pdb = entry["pdb"]
            sequence = entry["seq"]
            cdr = entry['cdr']
            cdr = cdr.replace("2", "T")

            yield id_, {"Key": pdb, "Sequence": sequence, "CDR": cdr}