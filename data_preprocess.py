import pandas as pd
import os
import glob
import numpy as np
from pathlib import Path
from utils import FileConfig

"""
Each data source will have its own configuration, including
    - file name or path to directory
    - sentence column name
    - label column name
    - delimitor character if source is csv
    - encoding (default to utf-8)
    - quoted vs not quoted 
"""


class TextData(object):
    pass


class TextClassifierData(TextData):
    """
     Data for text classification
    """

    @property
    def sentences(self):
        return self.data_df[self.file_config.sentence_column]

    @property
    def labels(self):
        return self.data_df[self.file_config.target_column]

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
                header=self.file_config.header,
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
            df[self.file_config.sentence_column].str.len() < self.file_config.MIN_SENTENCE_LENGTH)[0],
                axis=0,
                inplace=True
                )
        self.data_df = df.dropna(how='any', axis=0)

    def preprocess(self, **kwargs):
        """
        
        :param kwargs: 
        :return: 
        """
        self._load_data()
        self._clean_sentence_column()
        self._clean_label_column()
        self._drop_underrepresented_classes()
        
        
file_config = FileConfig(
    path_to_directory=Path('./data/train'),
    target_column='label',
    sequence_column='sentence',
    column_names=['index', 'label', 'sentence']
)


text_clas_data = TextClassifierData(file_config)
