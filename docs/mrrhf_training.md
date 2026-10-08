# MRRHF Training Guide

This guide provides step-by-step instructions for performing mrrhf training.

## Prerequisites

1. **Environment Setup**  
    Follow the [SFT Training Guide](./sft_training.md#environment-setup) to setup environment and activate it:
    ```bash
    conda activate JarvisIR
    ```
2. **SFT Training**  
    Follow the [SFT Training Guide](./sft_training.md) for supervised fine-tuning training to obtain the initial weights of pre-training.

3. **Generate Offline Data**  
    ​​Use [python script](../src/mrrhf/generate_offline_data.py) to sample offline data with weights from SFT training.​
    ```bash
    python src/mrrhf/generate_offline_data.py --save_path Offline_data.json --data_path rrhf_train.json
    ```
4. **File Replacement**  
    Use the file in 'dependences/qalign' to replace the qalign configuration file downloaded from huggingface to solve the library version compatibility issue.
    Then point the reward model to that local directory (used by `package/agent_tools/iqa_reward.py`, also needed by `inference.py` and `demo_gradio.py`):
    ```bash
    export QALIGN_PATH=/path/to/one-align
    ```

    
## MRRHF Training
Run the [training script](../train.sh) to perform mrrhf training. Alternatively, resume from our MRRHF pre-trained model [weights](https://huggingface.co/LYL1015/JarvisIR/tree/main/pretrained/mrrhf).
```bash
train.sh -c Model_Path -i Image_folder_path -d Data_files -o Offline_data -p Output_Path 
```


