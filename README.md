# Clinical Text Analysis

Repository for the fine-tuning and evaluation of the _med-gte-hybrid_ model. _Med-gte-hybrid_ is a fine-tuned sentence-transformer to embed clinical text achieving superior results in clinical NLP tasks. For training and evaluation we used MIMIC-IV v2.2 dataset which is why we are not allowed to publish evaluation datasets or training data. For access to the MIMIC-IV dataset refer to [PhysioNet](https://physionet.org/content/mimiciv/2.2/).

## Usage

The `/data/` folder expects the folders `hosp/`, `icu/` and `note/` from MIMIC-IV (v2.2) dataset as well as a `finetuning/` folder that holds sentences for fine-tuning in a .txt file.

* Run the fine-tuning using the script: <br>
`finetuning.py --model [folder of base model] --objective [simcse OR tsdae] --epochs [num epochs] --batch_size [batch size] --learning-rate [learning rate] --filepath [folder to training sentences file]`, this will store _med-gte-simcse/tsdae_ to `/models/`

* `mteb-eval.py` can be used to run MTEB evaluation with a local or HF model as input

* To run the mortality prediction, load the patient cohort into the `eval_datasets` folder, then use `precompute_embeddings.py` to compute embeddings of the cohort using the desired model. The embeddings can the be used in `mortality_pred.py` to run the mortality prediction.

* To run eGFR prediction and CKD prognosis, similarly, load the CKD cohort into the `eval_datasets` folder. Then refer to `compute_ckd_embeddings.ipynb` to prepare the embeddings for the downstream tasks. With these embeddings, tasks can then be run in `ckd_eval.ipynb`, the notebook contains necessary explanatory comments. 


Preprint: https://arxiv.org/abs/2502.15996
Cite our work using:
Kumar, A., Rauch, S., Cypko, M. and Amft, O., 2025. Med-gte-hybrid: A contextual embedding transformer model for extracting actionable information from clinical texts. arXiv preprint arXiv:2502.15996.
or bibtex: @article{kumar2025med,
  title={Med-gte-hybrid: A contextual embedding transformer model for extracting actionable information from clinical texts},
  author={Kumar, Aditya and Rauch, Simon and Cypko, Mario and Amft, Oliver},
  journal={arXiv preprint arXiv:2502.15996},
  year={2025}
}
