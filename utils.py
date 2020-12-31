class FileConfig:
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
        self.path_to_directory = path_to_directory,
        self.target_column = target_column,
        self.sequence_column = sequence_column,
        self.column_names = column_names or [sequence_column, target_column],
        self.delimiter = delimiter,
        self.header_column = header_column,
        self.encoding = encoding,
        self.min_sentence_length = min_sentence_length
