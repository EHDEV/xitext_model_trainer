from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast


def train_valid_test_split(sentences, labels, valid_pct=.15, test_pct=None):
    _texts, test_texts, _labels, test_labels = train_test_split(sentences, labels, test_size=test_pct)

    if not test_pct:
        return (_texts, test_texts), (_labels, test_labels)

    train_texts, val_texts, train_labels, val_labels = train_test_split(_texts, _labels, test_size=valid_pct)

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
