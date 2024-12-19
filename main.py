


from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from sklearn.model_selection import train_test_split
import wandb
import torch
class GPT2TrainerHF:
    def __init__(self, model_name="gpt2", project_name="GPT2-HF-Training"):
        # Initialisation
        self.model_name = model_name
        self.project_name = project_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Charger le modèle et le tokenizer depuis Hugging Face
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)

        # Ajouter un token de padding si absent
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialisation de Weights & Biases
        wandb.init(project=self.project_name)

    def prepare_data(self, file_path, block_size=128, eval_size=0.1):
        # Charger les données depuis un fichier texte local
        with open(file_path, "r", encoding="latin1") as f:
            lines = f.readlines()

        # Diviser les données en ensembles d'entraînement et d'évaluation
        train_texts, eval_texts = train_test_split(lines, test_size=eval_size, random_state=42)

        # Créer des datasets Hugging Face
        train_dataset = Dataset.from_dict({"text": train_texts})
        eval_dataset = Dataset.from_dict({"text": eval_texts})

        # Fonction de tokenisation
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=block_size
            )

        # Appliquer la tokenisation
        tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        return tokenized_train, tokenized_eval

    def setup_training_args(self, output_dir="results", epochs=3, batch_size=8):
        return TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_steps=500,
            eval_steps=500,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_total_limit=3,
            report_to="wandb",
            load_best_model_at_end=True
        )

    def train(self, train_dataset, eval_dataset, training_args):
        # Créer un Data Collator pour le masquage dynamique
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Pas de masquage aléatoire pour GPT-2
        )

        # Créer le Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Lancer l'entraînement
        trainer.train()

        # Terminer la session W&B
        wandb.finish()

# Script principal
if __name__ == "__main__":
    # Chemin vers le fichier texte
    dataset_path = "/content/rakitra"  # Modifiez en fonction du fichier extrait

    # Initialisation du formateur
    trainer = GPT2TrainerHF(model_name="gpt2", project_name="GPT2-HF")

    # Préparation des données
    tokenized_train, tokenized_eval = trainer.prepare_data(dataset_path, block_size=128)

    # Configuration des arguments d'entraînement
    training_args = trainer.setup_training_args(output_dir="gpt2_results", epochs=3, batch_size=8)

    # Entraînement du modèle
    trainer.train(tokenized_train, tokenized_eval, training_args)

