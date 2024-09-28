import mteb
from sentence_transformers import SentenceTransformer, models

import argparse 

from models import HybridModel

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='./models/med_gte_simcse', help='sentence transformer name to get from huggingface or local folder')

    args = parser.parse_args()

    # for med-gte-hybrid
    if args.model == 'med-gte-hybrid':
        model = HybridModel().to(device)
    else:
        model = SentenceTransformer(args.model, trust_remote_code=True).to(device)
    
    model.eval()

    ###
    # BIOSSES: Biomedical Semantic Similarity Estimation.
    # MedrxivClusteringP2P.v2: Clustering of titles+abstract from medrxiv. Clustering of 10 sets, based on the main category.
    # MedrxivClusteringS2S.v2: Clustering of titles from medrxiv. Clustering of 10 sets, based on the main category.
    # PublicHealthQA, MedicalQARetrieval: Retrieval of matching paragraphs to healthcare related questions
    ###
    
    tasks = mteb.get_tasks(languages=['eng'],
                            tasks=[
                                "BIOSSES",
                                "MedrxivClusteringS2S.v2", 
                                "MedrxivClusteringP2P.v2",
                                "MedicalQARetrieval",
                                "PublicHealthQA",
                                ])
    evaluation = mteb.MTEB(tasks=tasks)

    # Model needs a revision not None else mteb lib throws an error
    if not hasattr(model, 'revision') or model.revision is None:
        model.revision = "default_revision"    
        
    results = evaluation.run(model, output_folder="./results", overwrite_results=True)

    for result in results:
        name = dict(result)['task_name'] 
        if  name == 'BIOSSES':
            sp = dict(result)['scores']['test'][0]['cos_sim']['spearman']
            pe = dict(result)['scores']['test'][0]['cos_sim']['pearson']
            print(f"BIOSSES cos_sim | Spearman: {round(100*sp, 2)}, Pearson: {round(100*pe, 2)}")
        if name == "MedrxivClusteringP2P.v2" or name == "MedrxivClusteringS2S.v2" or name == "PublicHealthQA" or name == "MedicalQARetrieval":
            v = dict(result)['scores']['test'][0]['main_score']
            print(f"{name} | main score: {round(100*v,2)}")