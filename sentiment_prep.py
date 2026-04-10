#!/usr/bin/env python3
"""
Data preparation for directed financial sentiment extraction using MLX-LM
"""

import os
import json
import random
import yaml
import argparse
from typing import List, Dict

# First, install required dependencies
def install_dependencies():
    """Install required packages"""
    import subprocess
    import sys
    
    packages = [
        "mlx-lm",
        "datasets",
        "transformers",
        "torch",
        "pyyaml"
    ]
    
    # for package in packages:
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    subprocess.check_call(["uv", "add", " ".join(packages)])


class SentimentDataProcessor:
    """Process directed financial sentiment data for MLX-LM fine-tuning"""

    def __init__(self, dataset_path: str, seed: int = 42):

        self.system_prompt = (
            "You are a financial analyst specializing in directed sentiment extraction. "
            "Given a financial news text, identify all mentioned entities and determine "
            "the sentiment directed toward each one. Return your answer as a JSON array "
            "where each element has: \"entity\" (name), \"polarity\" (+ positive, - negative, "
            "0 neutral, ~ context-dependent), and \"category\" (one of: Legal, Business, "
            "Performance, Recruitment, NewsRelease, Bankruptcy)."
        )

        self.seed = seed
        self.records = self._load(dataset_path)

    def _load(self, path: str) -> List[Dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def create_chat_format(self, record: Dict) -> Dict:
        """Convert a single record to mlx-lm chat format."""
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Extract the directed financial sentiment from the following text:\n\n{record['text']}",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(record["extractions"], ensure_ascii=False),
                },
            ]
        }

    def save_for_mlx_training(self, output_dir: str = "data", split: float = 0.8):
        """Shuffle, split, and save processed data in MLX-LM compatible format."""
        os.makedirs(output_dir, exist_ok=True)

        random.seed(self.seed)
        shuffled = list(self.records)
        random.shuffle(shuffled)

        processed_data = [self.create_chat_format(r) for r in shuffled]

        # Split data into train/validation
        split_idx = int(len(processed_data) * split)
        train_data = processed_data[:split_idx]
        valid_data = processed_data[split_idx:]

        # Save as JSONL files
        train_path = os.path.join(output_dir, "train.jsonl")
        valid_path = os.path.join(output_dir, "valid.jsonl")

        for path, data in [(train_path, train_data), (valid_path, valid_data)]:
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Saved {len(train_data)} training samples to {train_path}")
        print(f"Saved {len(valid_data)} validation samples to {valid_path}")

        return train_path, valid_path


def create_training_config(model_name:str="mlx-community/Llama-3.2-3B-Instruct-4bit", iters:int=500, learning_rate:float=1e-5):
    """Create training configuration for MLX-LM"""
    config = {
        "model": model_name,  # The path to the local model directory or Hugging Face repo
        "train": True,  # Whether or not to train (boolean)
        "fine_tune_type": "lora",  # The fine-tuning method: "lora", "dora", or "full"
        "data": "data",  # Directory with {train, valid, test}.jsonl files
        "seed": 42,   # The PRNG seed
        "num_layers": 16,  # Number of layers to fine-tune (16 is a good balance of capacity vs. memory on 16 GB)
        "batch_size": 2, # Minibatch size (conservative for Llama 8B on 16 GB)
        "iters": iters,  # Iterations to train for
        "val_batches": 25,   # Number of validation batches, -1 uses the entire validation set
        "learning_rate": learning_rate, # Adam learning rate
        # "wand": "wandb-project" Whether to report the logs to WandB
        "steps_per_report": 10, # Number of training steps between loss reporting
        "steps_per_eval": 100,  # Number of training steps between validations
        # "resume_adapter_file": None,  # Load path to resume training with the given adapter weights
        "adapter_path": "adapters", # Save/load path for the trained adapter weights
        "save_every": 100,  # Save the model every N iterations
        "test": False,  # Evaluate on the test set after training
        "test_batches": 100,    # Number of test set batches, -1 uses the entire test set
        "max_seq_length": 512, # Maximum sequence length
        "grad_checkpoint": True,    # Use gradient checkpointing to reduce memory use
        "lora_parameters": {  # LoRA parameters can only be specified in a config file
            "keys": ["self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj"],   # q_proj + v_proj: standard minimal set k_proj + o_proj: added for better quality on nuanced polarity reasoning at modest extra memory cost
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.05
        }
    }
    
    with open('training_configs.yml', 'w') as f:
        yaml.dump(config, f, indent=2)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Prepare sentiment data for MLX-LM fine-tuning"
    )
    parser.add_argument(
        "--dataset_path",
        default="data/sentiment_training.jsonl",
        help="Path to the input JSONL dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Output directory for train.jsonl and valid.jsonl",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train fraction (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling",
    )

    parser.add_argument(
        "--model_name",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="LLM model to use"
    )
    # parser.add_argument(
    #     "--model_name",
    #     default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    #     help="LLM model to use"
    # )
    parser.add_argument(
        "--iters",
        default=500,
        help="Training iterations"
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        help="Training learning rate"
    )
    parser.add_argument(
        "--install_deps",
        action="store_true",
        help="Install required dependencies"
    )

    args = parser.parse_args()

    if args.install_deps:
        print("Installing dependencies...")
        install_dependencies()

    # Process the dataset
    print(f"Loading dataset from: {args.dataset_path}")
    processor = SentimentDataProcessor(args.dataset_path, seed=args.seed)
    print(f"Loaded {len(processor.records)} records")

    train_path, valid_path = processor.save_for_mlx_training(output_dir=args.output_dir, split=args.split)

    # Create training configuration
    print("Creating training configuration...")
    config = create_training_config(model_name=args.model_name, iters=args.iters, learning_rate=args.learning_rate)

    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print(f"✓ Dataset processed: {train_path}, {valid_path}")
    print(f"✓ Configuration saved: training_configs.yml")

    print("\nNext steps:")
    print("1. Fine-tune the model:")
    print("   [uv run] python sentiment_training.py")
    print("\n2. Or directly: mlx_lm.lora --config training_configs.yml")
    print("2. Run inference:")
    print('   [uv run] python sentiment_inf.py --text "Apple sued Samsung over patent violations."')


if __name__ == "__main__":
    main()
