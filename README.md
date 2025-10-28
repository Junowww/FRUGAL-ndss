# FRUGAL

## Project Structure

```
FRUGAL-master/
├── config.py                 # Configuration parameters
├── dqn_train_sac.py         # Main SAC training script
├── cw_df_test_sac.py        # Testing and evaluation script
├── sac_defense.py           # SAC agent implementation
├── network_SAC.py           # SAC neural networks
├── enviroment.py            # Training environment
├── reply_buffer.py          # Experience replay buffer
├── utility.py               # Utility functions
├── models/                  # Attack model implementations
│   ├── DF.py               
│   ├── TF.py               
│   ├── AWF.py              
│   ├── NetCLR.py           
│   └── VarCNN.py           
├── dataset/                 # Dataset files
├── checkpoint/              # Training checkpoints
└── saved_trained_models/    # Pre-trained models
```

## Installation

conda env create -f mut_info.yaml

## Quick Start

### 1. Basic SAC Defense Training

Train SAC defense against Deep Fingerprinting attacks:

```bash
python dqn_train_sac.py --subdir date-train --attack_model DF --bwo_para 0.3 
```

### 2. Testing Defense Effectiveness

Evaluate the trained defense model:

```bash
python cw_df_test_sac.py --attack_model DF --subdir date-test --bwo_para 0.3
```

## Configuration

### Training Parameters
- `--device`: Computing device (cpu, cuda:0, cuda:1)
- `--subdir`: Experiment name
- `--attack_model`: Target attack model (DF, TF, AWF, NetCLR, VarCNN)
- `--limits`: Number of training samples per class
- `--bwo_para`: Bandwidth overhead parameter 
- `--nb_classes`: Number of website classes
