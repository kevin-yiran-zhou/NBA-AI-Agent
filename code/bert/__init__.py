"""
BERT module for intent and attribute classification.
"""
from .bert import BertPredictor
from .train_bert import BertForIntentAndAttr

__all__ = ['BertPredictor', 'BertForIntentAndAttr']

