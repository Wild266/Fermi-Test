# download_models.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Download to local pretrained_models folder
cache_dir = "./pm"

# Download models
models = [
    "Qwen/Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" #deepseek-math-7b-base"
]

for model_name in models:
    print(f"Downloading {model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"Successfully downloaded {model_name}")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")