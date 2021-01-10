class FileConfig:
    """
    Data file configurations are updated here
    """
    def __init__(
            self,
            path_to_directory,
            target_column='label',
            sequence_column='text',
            column_names=None,
            delimiter=',',
            header_column=0,
            encoding='utf-8',
            min_sentence_length=5
    ):

        self.path_to_directory = path_to_directory
        self.target_column = target_column
        self.sequence_column = sequence_column
        self.column_names = column_names or [sequence_column, target_column]
        self.delimiter = delimiter
        self.header_column = header_column
        self.encoding = encoding
        self.min_sentence_length = min_sentence_length

#
# class ModelConfig:
#     def __init__(
#             self,
#             model_group,
#             pretrained_model_name='bert-base-uncased',
#             optimizer='adam',
#             scheduler='linear_with_warmup',
#             epochs=2,
#             seed=100,
#             evaluation_metric='accuracy',
#             output_dir='./models/',
#
#     ):
#         self.model_group = model_group
#         self.pretrained_model_name = model_group
