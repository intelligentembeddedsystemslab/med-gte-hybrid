import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from transformers import AutoModel
import nltk
import re
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_values = []

# Modifies loss functions to add logging of loss
class LoggingDenoisingAutoEncoderLoss(losses.DenoisingAutoEncoderLoss):
    def forward(self, sentence_features, labels):
        loss_value = super().forward(sentence_features, labels)

        loss_values.append(loss_value.item())

        return loss_value

class LoggingMultipleNegativesRankingLoss(losses.MultipleNegativesRankingLoss):
    def forward(self, sentence_features, labels):
        loss_value = super().forward(sentence_features, labels)

        loss_values.append(loss_value.item())

        return loss_value


def freeze_first_n_layers(model, n):
    # freezes first n encoder layers 
    # gte-large has 24 encoder blocks indexed 0 up to 23
    for name, param in model.named_parameters():
            if 'encoder.layer.' in name:
                layer_num = int(name.split('.')[4])
                if layer_num < n:
                    param.requires_grad = False


def save_loss():
    x_values = range(1, len(loss_values) + 1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, loss_values, marker='o', linestyle='-')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('Training Loss over Batches')
    plt.grid(True)
    plt.xticks(x_values)  # Show only integers on the x-axis

    # Show the plot
    plt.savefig("./plots/loss.png")
    plt.show()

def tsdae(model, sentences):
    # dataset class with noise functionality built-in
    train_data = DenoisingAutoEncoderDataset(sentences)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # finetunes model on the data using TSDAE
    loss = LoggingDenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler='warmupcosine',
        warmup_steps = (len(dataloader)*epochs) * 0.1,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True
    )
    save_loss()
    model.save('models/unsupervised_tsdae')
 

def simcse(model, sentences):
    # Convert train sentences to sentence pairs
    data = [InputExample(texts=[s, s]) for s in sentences]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # Use the MNRL loss
    train_loss = LoggingMultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler='warmupcosine',
        warmup_steps = (len(dataloader)*epochs) * 0.1,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True,
    )
    save_loss()
    model.save("models/unsupervised_simcse")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="thenlper/gte-large", help='sentence transformer name to get from huggingface')
    parser.add_argument('-o', '--objective', choices=['simcse', 'tsdae'], default='simcse', help='Choose between finetuning methods')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='What batch size to use')
    parser.add_argument('-lr', '--learning-rate', type=int, default=32, help='Learning rate')
    parser.add_argument('-f', '--filepath', default='./data/finetuning/sentences10k.txt', help="sentences filepath")

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    transformer = models.Transformer(args.model)
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), 'mean')
    model = SentenceTransformer(modules = [transformer, pooling]).to(device)

    freeze_first_n_layers(model, 21)

    sentences = []
        
    with open(args.filepath, 'r', encoding='utf-8') as file:
        # Read given file
        sentences = [sentence.strip() for sentence in file.readlines()]

    print(f'Loaded {len(sentences)} sentences')

    if args.objective == 'simcse':
        simcse(model, sentences)

    if args.objective == 'tsdae':
        tsdae(model, sentences)





