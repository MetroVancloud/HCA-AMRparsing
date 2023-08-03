# HCA-AMRparisng

This is the repo for [Addressing Long-Distance Dependencies in AMR Parsing with Hierarchical Clause Annotation](), which is based on the popular AMR parser, [SPRING](https://github.com/SapienzaNLP/spring).

## Data

### AMR
AMR 2.0, AMR 3.0, New3, Bio, TLP

### HCA features
located at `/data/hca_feature/`

## Installation
```shell script
cd spring
pip install -r requirements.txt
pip install -e .
```

The code only works with `transformers` < 3.0 because of a disrupting change in positional embeddings.
The code works fine with `torch` 1.5. We recommend the usage of a new `conda` env.

## Train
Modify `config.yaml` in `configs`.

```shell script
python bin/train.py --config configs/config.yaml --direction amr
```
Results in `runs/`

## Evaluate

```shell script
sh scripts/pred_amr.sh
```
`gold.amr.txt` and `pred.amr.txt` will contain, respectively, the concatenated gold and the predictions.

To reproduce our paper's results, you will also need to run the [BLINK](https://github.com/facebookresearch/BLINK) 
entity linking system on the prediction file (`data/tmp/amr2.0/pred.amr.txt` in the previous code snippet). 
To do so, you will need to install BLINK, and download their models:
```shell script
git clone https://github.com/facebookresearch/BLINK.git
cd BLINK
pip install -r requirements.txt
sh download_blink_models.sh
cd models
wget http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl
cd ../..
```
Then, you will be able to launch the `blinkify.py` script:
```shell
python bin/blinkify.py \
    --datasets data/tmp/amr2.0/pred.amr.txt \
    --out data/tmp/amr2.0/pred.amr.blinkified.txt \
    --device cuda \
    --blink-models-dir BLINK/models
```
To have comparable Smatch scores you will also need to use the scripts available at https://github.com/mdtux89/amr-evaluation, which provide
results that are around ~0.3 Smatch points lower than those returned by `bin/predict_amrs.py`.

## Linearizations
The previously shown commands assume the use of the DFS-based linearization. To use BFS or PENMAN decomment the relevant lines in `configs/config.yaml` (for training). As for the evaluation scripts, substitute the `--penman-linearization --use-pointer-tokens` line with `--use-pointer-tokens` for BFS or with `--penman-linearization` for PENMAN.
