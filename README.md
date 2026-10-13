# brain-lora

Code for *Scaling Parameter-Efficient Fine-Tuning for Brain Encoding*: a
LoRA-adapted Qwen3-Omni-30B-A3B as an encoding model of fMRI responses to
naturalistic video (CNeuroMod / Algonauts 2025).

## Method

* **Single-pass causal encoding.** Each 149 s training window (100 TRs) is
  encoded in one forward pass. Audio, video and transcript are interleaved in
  20 s chunks, and every token carries the stimulus time it describes. Because
  the backbone is causal, this gives a representation of each stream at every
  2 Hz step. A 60 s context lead precedes each window.
* **Adapter.** LoRA (rank 16, alpha 32) on the q/k/v/o projections of decoder
  layers 1-32. The decoder is truncated after layer 32, where the features are
  read. The audio and vision towers are frozen.
* **Readouts.**
  * *Linear* (52.3M parameters): a per-modality projection 2048 -> 512, an
    average over eight log-spaced lag blocks, and a linear map per subject to
    1000 Schaefer parcels.
  * *Cross-attention*: replaces the fixed lag average with per-TR attention
    over the LM token sequence. It is initialised to equal the lag average, so
    the same ridge solution applies.
* **Training.** LoRA and readout are trained jointly with AdamW for one epoch
  (1,429 windows, one window per step) on all four subjects. The readout starts
  from a closed-form ridge solution on frozen features.
* **Closed-form re-fit.** After training, the readout's linear map is re-solved
  by ridge regression on the adapted features. The penalty is chosen per parcel
  on held-out training windows.

## Results

Mean per-parcel Pearson r on Friends season 6 (validation), averaged over the
four subjects, after the re-fit. Each row is one run (seed 0).

| model | trainable parameters | r |
| --- | --- | --- |
| frozen backbone + linear | 52.3M | 0.291 |
| frozen backbone + cross-attention | 52.5M | 0.312 |
| LoRA + linear | 61.2M | 0.326 |
| LoRA + cross-attention | 61.4M | 0.331 |

## Setup

One 80 GB GPU (A100/A800/H100) per run, CUDA, Python >= 3.10.

```bash
pip install -e ".[test]"
pip install flash-attn --no-build-isolation
```

The backbone, `Qwen/Qwen3-Omni-30B-A3B-Thinking`, is downloaded from the
Hugging Face Hub on first use.

The paper's runs used torch 2.8-2.9 and transformers 4.57.1-4.57.3. Other
versions change the backbone's activations slightly (the mixture-of-experts
routing amplifies small numerical differences), so features are not
bit-identical across environments; scores on season 6 agree to about 1e-4 in r.

### Data

Download the Algonauts 2025 challenge data
([instructions](https://github.com/courtois-neuromod/algonauts_2025.competitors)),
then prepare it:

```bash
python -m brain_lora.cli.prepare_data --algonauts-root <download> --out data
```

This writes:

* `data/media_cache/`: every clip decoded once (2 Hz frames at 336x224,
  16 kHz audio, transcript), about 130 GB;
* `data/manifests/`: one row per subject, run and window. Training is Friends
  seasons 1-5 plus Movie10; validation is Friends season 6;
* `data/fmri.h5`: BOLD runs z-scored per parcel.

## Pretrained models

The two LoRA models of the table are on the Hugging Face Hub at
[`zfying/brain-lora`](https://huggingface.co/zfying/brain-lora)
(`lora_linear`, `lora_cross_attn`). Download one into a run directory and score
it:

```bash
python -m brain_lora.cli.hub download zfying/brain-lora lora_linear \
    --data-dir data --run-dir runs/lora_linear
python -m brain_lora.cli.evaluate runs/lora_linear
```

`python -m brain_lora.cli.hub export <run_dir> <out_dir>` writes a trained run
in the same layout: the LoRA adapter in PEFT format, the readout weights, and
the run configuration.

## Pipeline

```bash
# 1. readout initialisation: frozen features (30 s lead), PCA, ridge
python -m brain_lora.cli.extract_features --data-dir data --out data/features_L32_lead30.h5
python -m brain_lora.cli.fit_readout_init --data-dir data \
    --features data/features_L32_lead30.h5 --out data/readout_init_L32.npz

# 2. train LoRA + readout (~4.8 h on one A800)
python -m brain_lora.cli.train --data-dir data --readout-init data/readout_init_L32.npz \
    --run-dir runs/lora_linear

# 3. closed-form re-fit of the readout map, then score season 6
python -m brain_lora.cli.refit runs/lora_linear
python -m brain_lora.cli.evaluate runs/lora_linear --checkpoint refit_last.pt
```

Each training flag is documented in `--help`. The defaults are the paper's
recipe. `--readout cross_attn` selects the cross-attention readout, and
`--lora-lr 0` trains the readout on the frozen backbone (the frozen rows of the
table).

[`scripts/reproduce.sh`](scripts/reproduce.sh) lists the commands behind each
experiment:

| stage | experiment |
| --- | --- |
| `main` | frozen vs LoRA backbone x linear vs cross-attention readout |
| `ablations` | LoRA rank 32, gated-MLP readout, no audio input |
| `lead` | context lead of 0, 30, 60, 90 and 120 s |
| `seeds` | seed spread of the two LoRA models |
| `attribution` | modality attribution of the adapted and frozen models |
| `compare` | paired per-parcel comparison of two runs |

### Modality attribution

`brain_lora.analysis.attribution` computes exact Shapley values over the video,
audio and transcript inputs. An absent modality's LM input embeddings are
replaced by their mean over training windows, which keeps token counts and
positions. All eight modality subsets are evaluated on every validation window.
The report gives, per parcel and per region, each modality's share of the
prediction and its contribution to r, with episode block-bootstrap intervals.

### Comparing runs

`brain_lora.analysis.compare <control> <arm>` reports the mean per-parcel
difference in r with a 95% interval. Parcels are resampled in 100
functional-connectivity blocks.

## Layout

```
src/brain_lora/
  constants.py         clocks and window geometry
  metrics.py           per-parcel Pearson r
  training.py          run configuration, trainer, training loop
  hub.py               export and import of trained models (Hub layout)
  data/                stimulus decoding, manifests and fMRI, training windows
  model/
    inputs.py          single-pass input layout and token index
    encoder.py         Qwen3-Omni with LoRA, read at one layer
    readouts.py        linear, cross-attention and gated-MLP readouts
    refit.py           closed-form ridge with per-parcel penalties
  cli/                 prepare_data, extract_features, fit_readout_init,
                       train, refit, evaluate, hub
  analysis/            modality attribution, paired comparison
scripts/reproduce.sh   the paper's experiments
tests/                 unit tests (CPU)
```

## Tests

```bash
pytest tests
```

## Acknowledgements

The window geometry, fMRI preprocessing and train/validation split follow
[MIRAGE](https://github.com/epflneuroailab/mirage). The fMRI data are from the
Courtois NeuroMod project via the Algonauts 2025 challenge. The parcel labels
in `src/brain_lora/resources/` are derived from the Schaefer et al. (2018)
1000-parcel atlas.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
