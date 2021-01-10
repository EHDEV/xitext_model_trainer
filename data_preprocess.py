import pandas as pd
import os
import glob
import numpy as np
from pathlib import Path
from transformers import DistilBertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from config import FileConfig
from sklearn.model_selection import train_test_split

"""
Each data source will have its own configuration, including
    - file name or path to directory
    - sentence column name
    - label column name
    - delimitor character if source is csv
    - encoding (default to utf-8)
    - quoted vs not quoted 
"""


def _encode_text_into_tokens(sentences):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(
        sentences,
        add_special_tokens=True,
        max_length=64,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    return inputs


def train_valid_test_split(sentences, labels, valid_pct=.15, test_pct=None):
    _texts, test_texts, _labels, test_labels = train_test_split(sentences, labels, test_size=test_pct)

    if not test_pct:
        return (_texts, test_texts), (_labels, test_labels)

    train_texts, val_texts, train_labels, val_labels = train_test_split(_texts, _labels, test_size=valid_pct)

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def _wrap_tensors_in_dataloader(input_tensor, attention_mask_tensor, label_tensor, batch_size=32):
    # DataLoader requires batch_size for training
    # Recommended batch sizes are 16 and 32

    data_dts = TensorDataset(input_tensor, attention_mask_tensor, label_tensor)
    sampler = RandomSampler(data_dts)

    return DataLoader(data_dts, sampler=sampler, batch_size=batch_size)


class TextData(object):
    pass


class TextClassifierData(TextData):
    """
     Data for text classification
    """

    @property
    def sentences(self):
        return self.data_df[self.file_config.sentence_column].tolist()

    @property
    def labels(self):
        return self.data_df[self.file_config.target_column].tolist()

    @property
    def label_to_idx(self):
        classes = sorted(set(self.data_df[self.file_config.target_column]))
        return {
            label: i for i, label in enumerate(classes)
        }

    @property
    def label_tokens(self):
        return [
            self.label_to_idx[lb_text] for lb_text in self.labels
        ]

    @property
    def num_labels(self):
        return len(set(self.labels))

    def __init__(self, config_object):
        self.file_config = config_object

    def _load_data(self):
        file_path = self.file_config.path_to_directory
        if os.path.isdir(file_path):
            files = glob.glob(f'{file_path}/*.*')
        else:
            files = [file_path]

        dfs = [
            pd.read_csv(
                fp,
                sep=self.file_config.delimiter,
                names=self.file_config.column_names,
                header=self.file_config.header_column,
                encoding=self.file_config.encoding
            ) for fp in files
        ]
        self.data_df = pd.concat(dfs, axis=0, ignore_index=True)

    def _clean_label_column(self):
        """
        Remove trailing and beginning white characters and drop rows with empty labels
        :return: None
        """
        temp_df = self.data_df
        temp_df[self.file_config.target_column] = temp_df[self.file_config.target_column].str.lower().str.strip()
        temp_df = temp_df[temp_df[self.file_config.target_column] != '']

        self.data_df = temp_df

    def _drop_underrepresented_classes(self, threshhold=.05):
        zdf = pd.DataFrame(
            self.data_df[self.file_config.target_column].value_counts() / self.data_df.shape[0] > threshhold
        )
        valid_rows = zdf[zdf.label == True].index.tolist()

        self.data_df = self.data_df[(self.data_df.label.isin(valid_rows))]

    def _clean_sentence_column(self):
        df = self.data_df
        df[self.file_config.sentence_column] = df[self.file_config.sentence_column].str.lower().str.strip()
        df.drop(index=np.where(
            df[self.file_config.sentence_column].str.len() < self.file_config.min_sentence_length)[0],
                axis=0,
                inplace=True
                )
        self.data_df = df.dropna(how='any', axis=0)

    def _prepare_data_for_training(self):
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = train_valid_test_split(
            sentences=self.sentences, labels=self.label_tokens, valid_pct=0.20, test_pct=0.05
        )

        train_encodings = _encode_text_into_tokens(train_texts)
        val_encodings = _encode_text_into_tokens(val_texts)
        # test_encodings = _encode_text_into_tokens(test_texts)

        train_data_loader = _wrap_tensors_in_dataloader(
            input_tensor=train_encodings['input_ids'],
            attention_mask_tensor=train_encodings['attention_mask'],
            label_tensor=torch.tensor(train_labels)
        )
        val_data_loader = _wrap_tensors_in_dataloader(
            input_tensor=val_encodings['input_ids'],
            attention_mask_tensor=val_encodings['attention_mask'],
            label_tensor=torch.tensor(val_labels)
        )

        # test_data_loader = _wrap_tensors_in_dataloader(
        #     input_tensor=test_encodings['input_ids'],
        #     attention_mask_tensor=test_encodings['attention_mask'],
        #     label_tensor=test_labels
        # )

        return train_data_loader, val_data_loader

    def preprocess(self, **kwargs):
        """
        Cleanup and transform text data to make it suitable for classification
        :param kwargs:
        :return:
        """

        self._load_data()
        self._clean_sentence_column()
        self._clean_label_column()
        self._drop_underrepresented_classes()
        train_dataloader, val_dataloader = self._prepare_data_for_training()

        return train_dataloader, val_dataloader


file_config = FileConfig(
    path_to_directory=Path('/content/drive/MyDrive/elite/elitelearn/data/device_train.csv'),
    target_column='label',
    sequence_column='sentence',
    column_names=['index', 'label', 'sentence']
)

def execute():
    text_clas_data = TextClassifierData(file_config)
    return text_clas_data.preprocess()

    # train_torch_data, val_torch_data = text_clas_data.preprocess()


