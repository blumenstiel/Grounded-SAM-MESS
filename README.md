# Multi-domain Evaluation of Semantic Segmentation (MESS) with Grounded-SAM

[[Website (soon)](https://github.io)] [[arXiv (soon)](https://arxiv.org/)] [[GitHub](https://github.com/blumenstiel/MESS)]

This directory contains the code for the MESS evaluation of Grounded-SAM (Grounding DINO and SAM).

## Model

Grounded-SAM is developed for open-vocabulary instance segmentation by combining Grounding DINO with the Segment Anything Model (SAM) (see https://github.com/IDEA-Research/Grounded-Segment-Anything). 
SAM predicts a mask for each detected bounding box from Grounding DINO.
Next, the model combines all predicted segments of each class to perform semantic segmentation instead of instance segmentation. 
Pixels without positive predictions are assigned to the background class.

The Grounded-SAM implementation is based on https://github.com/luca-medeiros/lang-segment-anything/blob/main/lang_sam/lang_sam.py but uses the new Grounding DINO API to speed up the inference (Grounding DINO predicts multiple texts instead of a single caption, resulting in up to 10x faster inference). The inference time could be further reduced by using buffered text embeddings in Grounding DINO.

## Setup
Create a conda environment `groundedsam` and install the required packages. See [mess/README.md](mess/README.md) for details.
```sh
 bash mess/setup_env.sh
```

Prepare the datasets by following the instructions in [mess/DATASETS.md](mess/DATASETS.md). The `groundedsam` env can be used for the dataset preparation. If you evaluate multiple models with MESS, you can change the `dataset_dir` argument and the `DETECTRON2_DATASETS` environment variable to a common directory (see [mess/DATASETS.md](mess/DATASETS.md) and [mess/eval.sh](mess/eval.sh), e.g., `../mess_datasets`). 

The SAM weights are downloaded automatically.

## Evaluation
To evaluate the Grounded-SAM models on the MESS datasets, run
```sh
bash mess/eval.sh

# for evaluation in the background:
nohup bash mess/eval.sh > eval.log &
tail -f eval.log 
```

For evaluating a single dataset, select the DATASET from [mess/DATASETS.md](mess/DATASETS.md), the DETECTRON2_DATASETS path, and run
```
conda activate groundedsam
export DETECTRON2_DATASETS="datasets"
DATASET=<dataset_name>

# Grounded-SAM base
python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS vit_b OUTPUT_DIR output/Grounded-SAM_base/$DATASET
# Grounded-SAM large
python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS vit_l OUTPUT_DIR output/Grounded-SAM_large/$DATASET
# Grounded-SAM huge
python evaluate.py --eval-only --num-gpus 1 --config-file default_config.yaml DATASETS.TEST \(\"$DATASET\",\) MODEL.WEIGHTS vit_h OUTPUT_DIR output/Grounded-SAM_huge/$DATASET
```
