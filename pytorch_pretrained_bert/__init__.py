__version__ = "0.4.0"
from .tokenization import BertTokenizer, BasicTokenizer, WordpieceTokenizer
from .modeling import (BertConfig, BertModel, BertForPreTraining,
                       BertForMaskedLM, BertForNextSentencePrediction,
                       BertForSequenceClassification, BertForMultipleChoice,BertForQuestionAnsweringSpanMask,
                       BertForTokenClassification)
#与原pytorch_pretrained_bert比，原来的BertForQuestionAnswering改成BertForQuestionAnsweringSpanMask。
from .optimization import BertAdam
from .file_utils import PYTORCH_PRETRAINED_BERT_CACHE
