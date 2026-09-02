# Inference Guide

This guide provides instructions for running inference with JarvisIR to analyze degraded images and get restoration task recommendations.

## Prerequisites

### Environment Setup

Follow the [SFT Training Guide](./sft_training.md#environment-setup) to setup environment and activate it:
```bash
conda activate JarvisIR
```

### Model Weights

1. Create checkpoint directory: `mkdir -p ./checkpoints/`
2. use *Hugging Face CLI* to download checkpoints.
```bash
cd checkpoints
hf download LYL1015/JarvisIR --local-dir ./
mkdir -p q_align
cd q_align
hf download q-future/one-align --local-dir ./
```
3. Replace [path = None](../dependences/qalign/modeling_mplug_owl2.py) with the absolute path of ```q_align```. 
4. Point `IQAScore` at the ```q_align``` weights, either by passing `IQAScore(qalign_path=...)` or by exporting the environment variable before running:
   ```bash
   export QALIGN_PATH=/absolute/path/to/q_align
   ```
5. Replace ```checkpoints/q_align/modeling_llama2.py``` and ```checkpoints/q_align/modeling_mplug_owl2.py``` with ```dependences/qalign/modeling_llama2.py``` and ```dependences/qalign/modeling_mplug_owl2.py```.

## Usage

Basic inference command:

```bash
python inference.py \
    --from_checkpoint ./checkpoints/pretrained/ \
    --image_folder ./data/inference/images/ \
    --save_folder ./data/inference/results/
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--from_checkpoint` | str | **Required** | Path to model checkpoint directory |
| `--image_folder` | str | None | Folder containing images for inference |
| `--save_folder` | str | `data/inference/result` | Folder to save results |
| `--max_generation_length_of_sampling` | int | 384 | Maximum generation length |
| `--seed` | int | 42 | Random seed for reproducibility |

