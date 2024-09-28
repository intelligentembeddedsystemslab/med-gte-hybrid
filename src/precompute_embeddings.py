import torch
import pandas as pd
from tqdm import tqdm

import argparse

from models import modelsdict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='simcse', help='Model to use from src.models.py')
    parser.add_argument('-s', '--subjects', default='./eval_datasets/mortality.csv', help='csv file containing a subject_id column to include')
    parser.add_argument('-c', '--num_chunks', default=100, help='max number of text segments to embed for each subject')

    args = parser.parse_args()

    model = modelsdict[args.model]()
    model.model.eval()

    # load csv containing a 'subject_id' column
    subject_ids = pd.read_csv(args.subjects)

    # Load both dataframes at once using pandas.concat
    data = pd.concat([
        pd.read_csv("./data/note/discharge_processed.csv"),
        pd.read_csv("./data/note/radiology_processed.csv")
    ])

    # exclude all subject ids that are not in subject_ids
    data = data[data['subject_id'].isin(subject_ids['subject_id'])]

    # Group data by subject_id for efficient filtering
    grouped_data = data.groupby('subject_id')

    embedding_dict = {}
    chunk_limit = int(args.num_chunks) # cuts off text after [chunk_limit] chunks for each subject. 0 = no limit

    progress_bar = tqdm(grouped_data.groups.keys(), desc="Computing embeddings", unit="subject ID")

    for subject_id in progress_bar:
        # Get text chunks directly from the grouped dataframe
        text_chunks = grouped_data.get_group(subject_id)['text']
        if chunk_limit != 0:
            text_chunks = text_chunks.iloc[:chunk_limit]

        # get_emebdding expects a List[str]
        embedding_dict[subject_id] = torch.stack([torch.tensor(model.get_embedding([text])[0]) for text in text_chunks])


    subject_file = args.subjects.split('/')[-1].replace('.csv', '')
    # Save the embedding dictionary using torch.save
    torch.save(embedding_dict, f'./data/embeddings/{args.model}_{subject_file}.pt')