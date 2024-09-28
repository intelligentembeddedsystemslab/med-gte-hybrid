import torch
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, f1_score, average_precision_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import csv
import numpy as np
import random

from models import modelsdict


class SubjectDataset(Dataset):
    # Dataset based on subject ids
    def __init__(self, subjects, embeddings):
        self.subjects = subjects
        self.embeddings = embeddings

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        label = eval_dataset[eval_dataset['subject_id'] == subject_id]['died'].values[0]
        embedding = self.embeddings[subject_id]
        return subject_id, embedding, torch.tensor(label, dtype=torch.float32)


def train(model, dataloader, epochs=5, learning_rate=1e-3):
    
    model.train()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    total_loss = 0
    progress_bar = tqdm(range(epochs), desc="Training epochs")

    for epoch in progress_bar:
        total_loss = 0  

        for batch in dataloader:
            subject_id, embeddings_batch, labels_batch = batch
            
            outputs = model(embeddings_batch)
            loss = criterion(outputs.squeeze(), labels_batch.float())
            total_loss += loss.item() 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print average epoch loss after each epoch
        average_loss = total_loss / len(dataloader)
        progress_bar.set_postfix(loss=f"{average_loss:.4f}")


def evaluate(model, dataloader):
    model.eval()
    output_dict = {}
    labels_dict = {}
    
    # gather probabilities from classifier model 
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            subject_id, embeddings_batch, labels_batch = batch
            subject_id = subject_id.item()

            outputs = model(embeddings_batch).cpu().numpy()
            if subject_id in output_dict.keys():
                output_dict[subject_id].append(outputs)
            else:
                output_dict[subject_id] = [outputs]
                labels_dict[subject_id] = labels_batch.cpu().numpy()

    # Calc prediction for each subject: 
    labels = []
    probabilities = [] 
    predictions = []

    for subject_id, outputs in output_dict.items():
        label = labels_dict[subject_id]
        prob = np.mean(outputs)
        prediction = 1 if prob >= 0.5 else 0

        labels.append(label)
        probabilities.append(prob)
        predictions.append(prediction)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    auprc = average_precision_score(labels, probabilities)
    print(f'ROC AUC: {roc_auc*100:.2f}')
    print(f"AUPRC: {auprc*100:.2f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1: {f1 * 100:.2f}%')
    print('Confusion Matrix:')
    print(cm)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("./plots/classification_roc.png")
    plt.show()

    return roc_auc, auprc


def calc_avg_embeddings(embeddings):
    avg_embedding = {}
    # calc avg embedding
    for subject_id, subject_data in embeddings.items():

        mean_tensor = torch.mean(subject_data, dim=0)  # Mean along the 0th dimension
        # Add the mean tensor to the result dictionary
        avg_embedding[subject_id] = mean_tensor

    return avg_embedding


def cross_validation(model, dataset, num_folds=5, epochs=5, learning_rate=1e-3):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_results_roc_auc = []
    fold_results_auprc = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{num_folds}")
        
        # Reset the model for each fold
        model = modelsdict[args.model]()
        
        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)
        
        trainloader = DataLoader(train_subsampler, batch_size=32, shuffle=True)
        valloader = DataLoader(val_subsampler, batch_size=1, shuffle=False)
        
        # Train the model
        train(model, trainloader, epochs=epochs, learning_rate=learning_rate)
        
        # Evaluate the model
        roc_auc, auprc = evaluate(model, valloader)
        fold_results_roc_auc.append(roc_auc)
        fold_results_auprc.append(auprc)
        
        print(f"Fold {fold + 1} ROC AUC: {roc_auc:.4f}")
        print(f"Fold {fold + 1} AUPRC: {auprc:.4f}")
        print("-" * 50)
    
    print(f"Average ROC AUC across {num_folds} folds: {np.mean(fold_results_roc_auc):.4f} (+/- {np.std(fold_results_roc_auc):.4f})")
    print(f"Average AUPRC across {num_folds} folds: {np.mean(fold_results_auprc):.4f} (+/- {np.std(fold_results_auprc):.4f})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ed', '--eval-datasets', type=str, default="./eval_datasets/mortality.csv", help='Path to the csv file for prediction')
    parser.add_argument('-e', '--embeddings', type=str, default="./data/embeddings/hybrid_mortality.pt", help='Path to embeddings')
    
    parser.add_argument('-mod', '--model', default='clf', help='Model to use src.models.py')
    parser.add_argument('-o', '--option', choices=['EVAL', 'TRAIN'], default='TRAIN', help='Choose between EVAL and TRAIN modes')
    parser.add_argument('-ep', '--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('-l', '--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    model = modelsdict[args.model]()

    torch.manual_seed(1)
    random.seed(0)

    epochs = args.epochs
    lr = args.lr

    eval_dataset = pd.read_csv(args.eval_datasets)
    embeddings = torch.load(args.embeddings)

    subjects = []
    labels = []

    subject_ids = eval_dataset['subject_id'].values
    for subject_id in subject_ids:
        subjects.append(subject_id)
        labels.append(eval_dataset[eval_dataset['subject_id'] == subject_id]['died'].values[0])

    embeddings = calc_avg_embeddings(embeddings)        
    dataset = SubjectDataset(subjects, embeddings)
    
    # Perform 5-fold cross-validation
    cross_validation(model, dataset, num_folds=5, epochs=epochs, learning_rate=lr)