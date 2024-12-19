# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RqtoKo2C_PDIVnnIyQnABt_RmC9HbMiq
"""

from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_data(file_path, tokenizer, block_size=128, eval_size=0.1):
    # Charger les données depuis un fichier CSV
    data = pd.read_csv(file_path, encoding="latin1")
    texts = data["text"].tolist()

    # Diviser les données en ensembles d'entraînement et d'évaluation
    train_texts, eval_texts = train_test_split(texts, test_size=eval_size, random_state=42)

    # Créer des datasets Hugging Face
    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    # Fonction de tokenisation
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size
        )

    # Appliquer la tokenisation
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    return tokenized_train, tokenized_eval