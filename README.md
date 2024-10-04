# Bhasha Sutra

**Bhasha Sutra** is a GPT-like language model project aimed at developing a deep understanding of large language models (LLMs) by pretraining and finetuning a custom model from scratch using the PyTorch ecosystem. The project follows a step-by-step approach to explore foundational model concepts and implement a fully operational language model.

## Project Structure


## Key Features

- **GPT-like Model:** A custom-built transformer model similar to GPT for language generation tasks.
- **Pretraining and Finetuning:** Implements both pretraining from scratch and finetuning on custom datasets.

<!--
BhashaSutra/
│
├── datasets/                   # Datasets for pretraining and fine-tuning
│   ├── raw/                    # Raw unprocessed datasets
│   └── processed/              # Preprocessed data, tokenized and ready for training
│
├── src/                        # Core source code for the model
│   ├── model/                  # Model architecture files
│   │   ├── gpt_model.py        # Core GPT model implementation
│   │   └── layers.py           # Transformer blocks, attention, etc.
│   ├── training/               # Training scripts and utilities
│   │   ├── train.py            # Model training script
│   │   └── data_loader.py      # Data loading and batching utilities
│   └── evaluation/             # Scripts for evaluating model performance
│       └── evaluate.py         # Evaluation metrics like perplexity, accuracy
│
├── experiments/                # Logging and checkpoints for different model versions
│   ├── baseline/               # Experiment folder for the baseline model
│   └── improved/               # Experiment folder with improvements and hyperparameter tuning
│
├── explore/                # Logging and checkpoints for different model versions
│   ├── libraries/               # Experiment folder for the baseline model
│   └── / 
├── logs/                       # Training logs and metrics
│
├── notebooks/                  # Jupyter notebooks for exploratory work and experiments
│   ├── data_preprocessing.ipynb # Data preprocessing steps
│   ├── model_testing.ipynb      # Testing model outputs
│   └── pytorch_concepts.ipynb   # Exploring PyTorch core concepts (autograd, tensors, etc.)
│
├── config/                     # Configuration files for experiments
│   ├── config.yaml             # General configuration (batch size, learning rate, etc.)
│   └── model_config.yaml       # Model-specific configurations (layers, heads, dimensions)
│
├── scripts/                    # Helper scripts for automation
│   ├── download_data.py        # Script to download and process datasets
│   └── pretrain_model.py       # Script to automate the pretraining process
│
├── tests/                      # Unit tests for different components
│   ├── test_model.py           # Tests for model architecture
│   ├── test_data_loader.py     # Tests for data loading
│   └── test_training.py        # Tests for training scripts
│
├── docs/                       # Documentation files
│   ├── architecture.md         # Overview of the model architecture
│   ├── training_process.md     # Details on training strategies, optimization, etc.
│   └── setup_instructions.md   # Instructions for setting up the project environment
│
├── requirements.txt            # List of Python dependencies (e.g., PyTorch, Transformers)
├── README.md                   # Project overview and setup guide
└── LICENSE                     # License for the project


-->