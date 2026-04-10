#!/usr/bin/env python3
"""
Training script for directed financial sentiment extraction using MLX-LM LoRA
"""

import yaml
import os
import subprocess
import argparse
from pathlib import Path
from mlx_lm import load, generate


def run_mlx_training(config_path: str = "training_configs.yml"):
    """Run MLX-LM LoRA training"""

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        print("Training config:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    configs = {'model': config['model'], 'adapter_path': config['adapter_path'], 'test': config['test']}

    # Construct MLX-LM command
    cmd = [
        "mlx_lm.lora",
        "--config",
        config_path,
    ]

    print("\nStarting MLX-LM training:")
    print(" ".join(cmd))
    print("\n" + "="*50)

    # Run training
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print("Training completed successfully!")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        success = False

    return configs, success


def test_trained_model(model_name:str="mlx-community/Llama-3.2-3B-Instruct-4bit", adapter_path:str="adapters"):
    """Test the trained model with sample coreference resolution"""
    
    print(f"Loading model {model_name} with adapter {adapter_path}...")
    
    try:
        # Load the model with the trained adapter
        model, tokenizer = load(model_name, adapter_path=adapter_path)
        
        # Test samples
        test_samples = [
            "Microsoft announced its quarterly earnings yesterday. The tech giant reported strong growth in cloud services.",
            "Tesla unveiled its new electric vehicle model. The automotive company highlighted the car's innovative features.",
            "OpenAI launched its latest language model. The AI company demonstrated impressive capabilities."
        ]
        
        print("\n" + "="*50)
        print("TESTING TRAINED MODEL")
        print("="*50)
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\nTest {i}:")
            print(f"Input: {sample}")
            
            # Create prompt
            prompt = f"Resolve all coreferences in the following text by replacing pronouns and descriptive references with their original entities: {sample}"
            
            # Generate response
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt,
                # max_tokens=1000,
                # temp=0.1
                # verbose=True
            )
            
            print(f"Output: {response}")
            print("-" * 50)
    
    except Exception as e:
        print(f"Error testing model: {e}")
        print("Make sure the adapter has been trained and saved correctly.")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune directed financial sentiment extraction model"
    )
    parser.add_argument(
        "--config",
        default="training_configs.yml",
        help="Training configuration YAML file",
    )

    args = parser.parse_args()

    print("Starting directed financial sentiment training...")
    config, success = run_mlx_training(args.config)

    if success:
        adapter_path = config.get("adapter_path", "adapters")
        model = config.get("model", "mlx-community/Llama-3.2-3B-Instruct-4bit")

        print("\n" + "=" * 50)
        print("TRAINING COMPLETE!")
        print("=" * 50)
        print(f"Adapter saved to: {adapter_path}")
        print("\nUsage examples:")
        print(
            '[uv run] python sentiment_inf.py --text "Apple sued Samsung over patent violations."'
        )
        print(
            f"[uv run] python -m mlx_lm.generate --model {model} "
            f"--adapter-path {adapter_path} "
            '--prompt "Extract the directed financial sentiment from: Apple sued Samsung."'
        )
    else:
        print("\nTraining failed. Check the error messages above.")


if __name__ == "__main__":
    main()
