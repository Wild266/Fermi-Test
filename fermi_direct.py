# fermi_direct.py
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FermiDirectExperiment:
    def __init__(self, csv_path='LLN_Dataset.csv'):
        self.df = pd.read_csv(csv_path, encoding='windows-1252')
        self.results = []
        
        # Check available GPUs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        
        # Load models
        logger.info("Loading base model...")
        self.base_model, self.base_tokenizer = self.load_model("Qwen/Qwen-7B", device_id=0)
        
        logger.info("Loading reasoning model...")
        self.reasoning_model, self.reasoning_tokenizer = self.load_model(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
            device_id=1 if torch.cuda.device_count() > 1 else 0
        )
    
    def load_model(self, model_name, device_id=0):
        """Load model from cache"""
        cache_dir = "./pm"
        device = f"cuda:{3}"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    def generate(self, prompt, model, tokenizer, use_cot=False):
        """Generate response from model"""
        if use_cot:
            prompt = f"{prompt}\n\nLet me think step by step:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        
        return response
    
    def extract_power(self, text):
        """Extract power of ten from response"""
        import re
        patterns = [
            r'10\^(-?\d+)', 
            r'10\*\*(-?\d+)', 
            r'power of ten[:\s]+(-?\d+)',
            r'answer[:\s]+(-?\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return 0
        return None
    
    def run(self):
        """Run experiment"""
        for idx, row in self.df[:].iterrows():
            # if idx >= 10:  # Test first 10 questions
            #     break
            if row['Questions'] == '' or pd.isna(row['Questions']):
                continue
            question = row['Questions']
            # print(f'Question {idx+1}: {question}')
            true_ans = row['Nearest Power of ten']
            prompt = f"Provide the nearest power of ten: {question}\n\nAnswer:"
            
            logger.info(f"\nQuestion {idx+1}: {question[:60]}...")
            
            # Base model
            base_resp = self.generate(prompt, self.base_model, self.base_tokenizer)
            base_ans = self.extract_power(base_resp)
            logger.info(f"Base response: {base_resp[:100]}...")
            logger.info(f"Base answer: {base_ans}")
            
            # Reasoning model  
            reasoning_resp = self.generate(prompt, self.reasoning_model, self.reasoning_tokenizer, use_cot=True)
            reasoning_ans = self.extract_power(reasoning_resp)
            logger.info(f"Reasoning response: {reasoning_resp[:100]}...")
            logger.info(f"Reasoning answer: {reasoning_ans}")
            
            # Store results
            self.results.append({
                'question': question,
                'true': true_ans,
                'base_resp': base_resp[:],
                'base_ans': base_ans,
                'reasoning_resp': reasoning_resp[:],
                'reasoning_ans': reasoning_ans,
                'base_correct': base_ans == true_ans if base_ans else False,
                'reasoning_correct': reasoning_ans == true_ans if reasoning_ans else False
            })
            
            # Save progress
            pd.DataFrame(self.results).to_csv('results/direct_results.csv', index=False)
    
    def analyze(self):
        """Analyze results"""
        df = pd.DataFrame(self.results)
        
        base_acc = df['base_correct'].mean()
        reasoning_acc = df['reasoning_correct'].mean()
        
        print(f"\n=== RESULTS ({len(df)} questions) ===")
        print(f"Base accuracy: {base_acc:.1%}")
        print(f"Reasoning accuracy: {reasoning_acc:.1%}")
        print(f"Improvement: {reasoning_acc - base_acc:+.1%}")
        
        # Show some examples
        print("\nExample responses:")
        for i in range(min(3, len(df))):
            print(f"\nQ: {df.iloc[i]['question'][:80]}...")
            print(f"True: {df.iloc[i]['true']}")
            print(f"Base: {df.iloc[i]['base_ans']} - Response: {df.iloc[i]['base_resp'][:80]}...")
            print(f"Reasoning: {df.iloc[i]['reasoning_ans']} - Response: {df.iloc[i]['reasoning_resp'][:80]}...")

if __name__ == "__main__":
    exp = FermiDirectExperiment()
    exp.run()
    exp.analyze()