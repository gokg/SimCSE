## Reproducibility  for SimCSE: Simple Contrastive Learning of Sentence Embeddings (2021)

This repository contains the code, prepared test notebooks and reproducibility result of [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821). The reproducibility is done by Gökhan Güler (gulerg17@itu.edu.tr).

This repository is forked from the authors' code repository. The original repository can be found here: https://github.com/princeton-nlp/SimCSE

To train the model, you can use the notebook at [Notebooks/Train.ipynb](https://github.com/gokg/SimCSE/tree/main/Notebooks/Train.ipynb)

To evaluate the model, you can use the notebook at [Notebooks/Evaluate.ipynb](https://github.com/gokg/SimCSE/tree/main/Notebooks/Evaluate.ipynb)

Training results are contained at their respective notebooks at the folder [Notebooks/trainings](https://github.com/gokg/SimCSE/tree/main/Notebooks/trainings)

Test results are done in a local CPU, and the outputs are saved at [Results](https://github.com/gokg/SimCSE/tree/main/Results) folder.

## Model List

You can use the following pre-trained models.

* [princeton-nlp/unsup-simcse-bert-base-uncased](https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased) 
* [princeton-nlp/unsup-simcse-bert-large-uncased](https://huggingface.co/princeton-nlp/unsup-simcse-bert-large-uncased)  
* [princeton-nlp/unsup-simcse-roberta-base](https://huggingface.co/princeton-nlp/unsup-simcse-roberta-base)     
* [princeton-nlp/unsup-simcse-roberta-large](https://huggingface.co/princeton-nlp/unsup-simcse-roberta-large)    
* [princeton-nlp/sup-simcse-bert-base-uncased](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased)  
* [princeton-nlp/sup-simcse-bert-large-uncased](https://huggingface.co/princeton-nlp/sup-simcse-bert-large-uncased)
* [princeton-nlp/sup-simcse-roberta-base](https://huggingface.co/princeton-nlp/sup-simcse-roberta-base)
* [princeton-nlp/sup-simcse-roberta-large](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large)    


## Evaluation

**Arguments**

Arguments for the evaluation are as follows,

* `--model_name_or_path`: The name or path of a `transformers`-based pre-trained checkpoint. You can directly use the models in the above table, e.g., `princeton-nlp/sup-simcse-bert-base-uncased`.
* `--pooler`: Pooling method. Now we support
    * `cls` (default): Use the representation of `[CLS]` token. A linear+activation layer is applied after the representation (it's in the standard BERT implementation). If you use **supervised SimCSE**, you should use this option.
    * `cls_before_pooler`: Use the representation of `[CLS]` token without the extra linear+activation. If you use **unsupervised SimCSE**, you should take this option.
    * `avg`: Average embeddings of the last layer. If you use checkpoints of SBERT/SRoBERTa ([paper](https://arxiv.org/abs/1908.10084)), you should use this option.
    * `avg_top2`: Average embeddings of the last two layers.
    * `avg_first_last`: Average embeddings of the first and last layers. If you use vanilla BERT or RoBERTa, this works the best.
* `--mode`: Evaluation mode
    * `test` (default): The default test mode. To faithfully reproduce our results, you should use this option.
    * `dev`: Report the development set results. Note that in STS tasks, only `STS-B` and `SICK-R` have development sets, so we only report their numbers. It also takes a fast mode for transfer tasks, so the running time is much shorter than the `test` mode (though numbers are slightly lower).
    * `fasttest`: It is the same as `test`, but with a fast mode so the running time is much shorter, but the reported numbers may be lower (only for transfer tasks).
* `--task_set`: What set of tasks to evaluate on (if set, it will override `--tasks`)
    * `sts` (default): Evaluate on STS tasks, including `STS 12~16`, `STS-B` and `SICK-R`. This is the most commonly-used set of tasks to evaluate the quality of sentence embeddings.
    * `transfer`: Evaluate on transfer tasks.
    * `full`: Evaluate on both STS and transfer tasks.
    * `na`: Manually set tasks by `--tasks`.
* `--tasks`: Specify which dataset(s) to evaluate on. Will be overridden if `--task_set` is not `na`. See the code for a full list of tasks.

### Training

**Data**

For unsupervised SimCSE, we sample 1 million sentences from English Wikipedia; for supervised SimCSE, we use the SNLI and MNLI datasets. You can run `data/download_wiki.sh` and `data/download_nli.sh` to download the two datasets.

**Arguments**

* `--train_file`: Training file path. We support "txt" files (one line for one sentence) and "csv" files (2-column: pair data with no hard negative; 3-column: pair data with one corresponding hard negative instance). You can use our provided Wikipedia or NLI data, or you can use your own data with the same format.
* `--model_name_or_path`: Pre-trained checkpoints to start with. For now we support BERT-based models (`bert-base-uncased`, `bert-large-uncased`, etc.) and RoBERTa-based models (`RoBERTa-base`, `RoBERTa-large`, etc.).
* `--temp`: Temperature for the contrastive loss.
* `--pooler_type`: Pooling method. It's the same as the `--pooler_type` in the [evaluation part](#evaluation).
* `--mlp_only_train`: We have found that for unsupervised SimCSE, it works better to train the model with MLP layer but test the model without it. You should use this argument when training unsupervised SimCSE models.
* `--hard_negative_weight`: If using hard negatives (i.e., there are 3 columns in the training file), this is the logarithm of the weight. For example, if the weight is 1, then this argument should be set as 0 (default value).
* `--do_mlm`: Whether to use the MLM auxiliary objective. If True:
  * `--mlm_weight`: Weight for the MLM objective.
  * `--mlm_probability`: Masking rate for the MLM objective.


