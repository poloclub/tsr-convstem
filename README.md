# High-Performance Transformers for Table Structure Recognition Need Early Convolutions

Table structure recognition (TSR) aims to convert tabular images into a machine-readable format, where a visual encoder extracts image features and a textual decoder generates table-representing tokens. Existing approaches use classic convolutional neural network (CNN) backbones for the visual encoder and transformers for the textual decoder. However, this hybrid CNN-Transformer architecture introduces a complex visual encoder that accounts for nearly half of the total model parameters, markedly reduces both training and inference speed, and hinders the potential for self-supervised learning in TSR. In this work, we design a lightweight visual encoder for TSR without sacrificing expressive power. We discover that a convolutional stem can match classic CNN backbone performance, with a much simpler model. The convolutional stem strikes an optimal balance between two crucial factors for high-performance TSR: a higher receptive field (RF) ratio and a longer sequence length. This allows it to "see" an appropriate portion of the table and "store" the complex table structure within sufficient context length for the subsequent transformer. We conducted reproducible ablation studies and open-sourced our code to enhance transparency, inspire innovations, and facilitate fair comparisons in our domain as tables are a promising modality for representation learning.
<p align="center">
    <img src="imgs/pipeline.png" alt="drawing" width="600"/>
</p>

### Installation
1. Prepare PubTabNet dataset available [here](https://github.com/ibm-aur-nlp/PubTabNet/tree/master#getting-data)
2. Change the "pubtabnet_dir" in [Makefile](./Makefile) to "your path to PubTabNet"
3. Set up venv
```bash
make .venv_done
```

### Training, Testing & Evaluation
1. Train an instance of visual encoder with ResNet-18
```bash
make experiments/r18_e2_d4_adamw/.done_train_structure
```
2. Test + Compute teds score
```bash
make experiments/r18_e2_d4_adamw/.done_teds_structure
```
3. All models in ablations are defined in "Experiment Configurations" section of [Makefile](./Makefile). Replace "r18_e2_d4_adamw" with any other configuration for training and testing.

