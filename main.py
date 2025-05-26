#!/usr/bin/env python3
"""
Neural Small Language Model (SLM) Inference CLI
Interactive command-line interface for text generation using trained SLM.
"""

import pickle
import sys
from typing import List, Tuple, Optional, Any

import numpy as np


class ColorPrinter:
    """Utility class for colored terminal output."""
    
    COLORS = {
        'YELLOW': '\033[93m',   # Prompts
        'GREEN': '\033[92m',    # Answers
        'MAGENTA': '\033[95m',  # Confidence
        'RED': '\033[91m',      # Errors
        'BLUE': '\033[94m',     # Info
        'RESET': '\033[0m'      # Reset
    }
    
    @classmethod
    def print_colored(cls, text: str, color: str) -> None:
        """Print text in specified color."""
        color_code = cls.COLORS.get(color.upper(), cls.COLORS['RESET'])
        print(f"{color_code}{text}{cls.COLORS['RESET']}")
    
    @classmethod
    def print_prompt(cls, text: str) -> None:
        """Print user prompt in yellow."""
        cls.print_colored(f"🤔 Prompt: {text}", 'YELLOW')
    
    @classmethod
    def print_answer(cls, text: str) -> None:
        """Print model answer in green."""
        cls.print_colored(f"🤖 Answer: {text}", 'GREEN')
    
    @classmethod
    def print_confidence(cls, confidence: float) -> None:
        """Print confidence score in magenta."""
        cls.print_colored(f"📊 Confidence: {confidence:.2%}", 'MAGENTA')
    
    @classmethod
    def print_error(cls, text: str) -> None:
        """Print error message in red."""
        cls.print_colored(f"❌ Error: {text}", 'RED')
    
    @classmethod
    def print_info(cls, text: str) -> None:
        """Print info message in blue."""
        cls.print_colored(f"ℹ️  {text}", 'BLUE')


class TextGenerator:
    """Text generation engine using trained Neural SLM."""
    
    def __init__(self, model: Any, tokenizer: Any, confidence_threshold: float = 0.3):
        self.model = model
        self.tokenizer = tokenizer
        self.confidence_threshold = confidence_threshold
        self.fallback_response = "Maaf, saya belum tahu."
    
    def preprocess_input(self, text: str) -> str:
        """Preprocess user input text."""
        # Basic cleaning similar to training preprocessing
        import re
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s\.,!?;:\-\'\"]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Apply softmax with temperature scaling."""
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def sample_token(self, probs: np.ndarray, top_k: int = 50, top_p: float = 0.9) -> Tuple[int, float]:
        """Sample next token using top-k and top-p sampling."""
        # Top-k sampling
        if top_k > 0:
            top_k_indices = np.argsort(probs)[-top_k:]
            top_k_probs = probs[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs)
        else:
            top_k_indices = np.arange(len(probs))
            top_k_probs = probs
        
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_indices = np.argsort(top_k_probs)[::-1]
            sorted_probs = top_k_probs[sorted_indices]
            cumsum_probs = np.cumsum(sorted_probs)
            
            # Find cutoff index
            cutoff_idx = np.searchsorted(cumsum_probs, top_p) + 1
            cutoff_idx = min(cutoff_idx, len(sorted_probs))
            
            # Keep only top-p tokens
            nucleus_indices = sorted_indices[:cutoff_idx]
            nucleus_probs = sorted_probs[:cutoff_idx]
            nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
            
            # Sample from nucleus
            sample_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
            token_idx = top_k_indices[nucleus_indices[sample_idx]]
            confidence = nucleus_probs[sample_idx]
        else:
            # Sample from top-k
            sample_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
            token_idx = top_k_indices[sample_idx]
            confidence = top_k_probs[sample_idx]
        
        return token_idx, float(confidence)
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.8, top_k: int = 50, 
                     top_p: float = 0.9) -> Tuple[str, float]:
        """Generate text continuation for given prompt."""
        try:
            # Preprocess and encode prompt
            processed_prompt = self.preprocess_input(prompt)
            input_tokens = self.tokenizer.encode(processed_prompt)
            
            # Initialize generation
            generated_tokens = input_tokens.copy()
            confidences = []
            
            # Get special token IDs
            eos_token = self.tokenizer.vocab.get('<EOS>', -1)
            pad_token = self.tokenizer.vocab.get('<PAD>', 0)
            
            # Generate tokens autoregressively
            for _ in range(max_length):
                # Prepare input (take last seq_len tokens)
                if len(generated_tokens) > self.model.max_seq_len:
                    input_seq = generated_tokens[-self.model.max_seq_len:]
                else:
                    input_seq = generated_tokens
                
                # Pad if necessary
                if len(input_seq) < self.model.max_seq_len:
                    padding = [pad_token] * (self.model.max_seq_len - len(input_seq))
                    input_seq = padding + input_seq
                
                # Convert to numpy array and add batch dimension
                input_array = np.array([input_seq])
                
                # Forward pass
                logits = self.model.forward(input_array)
                
                # Get logits for last position
                last_logits = logits[0, -1, :]  # [vocab_size]
                
                # Apply softmax with temperature
                probs = self.softmax(last_logits, temperature)
                
                # Sample next token
                next_token, confidence = self.sample_token(probs, top_k, top_p)
                
                # Check for end of sequence
                if next_token == eos_token:
                    break
                
                # Add to generated sequence
                generated_tokens.append(next_token)
                confidences.append(confidence)
                
                # Safety check
                if len(generated_tokens) > len(input_tokens) + max_length:
                    break
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated_tokens)
            
            # Extract only the generated part (remove original prompt)
            original_text = self.tokenizer.decode(input_tokens)
            if generated_text.startswith(original_text):
                generated_text = generated_text[len(original_text):].strip()
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return generated_text, avg_confidence
            
        except Exception as e:
            ColorPrinter.print_error(f"Generation failed: {e}")
            return self.fallback_response, 0.0
    
    def is_confident(self, confidence: float) -> bool:
        """Check if model confidence is above threshold."""
        return confidence >= self.confidence_threshold


class SLMInterface:
    """Interactive CLI interface for the Neural SLM."""
    
    def __init__(self, model_path: str = 'slm.pkl'):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load trained model and tokenizer from pickle file."""
        try:
            ColorPrinter.print_info(f"Loading model from {self.model_path}...")
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.tokenizer = model_data['tokenizer']
            
            # Initialize text generator
            self.generator = TextGenerator(self.model, self.tokenizer)
            
            # Print model info
            config = model_data.get('config', {})
            ColorPrinter.print_info("Model loaded successfully!")
            ColorPrinter.print_info(f"Vocabulary size: {model_data.get('vocab_size', 'Unknown')}")
            ColorPrinter.print_info(f"Model dimension: {config.get('d_model', 'Unknown')}")
            ColorPrinter.print_info(f"Layers: {config.get('n_layers', 'Unknown')}")
            ColorPrinter.print_info(f"Attention heads: {config.get('n_heads', 'Unknown')}")
            
        except FileNotFoundError:
            ColorPrinter.print_error(f"Model file not found: {self.model_path}")
            ColorPrinter.print_info("Please make sure you have trained the model using training.py")
            sys.exit(1)
        except Exception as e:
            ColorPrinter.print_error(f"Failed to load model: {e}")
            sys.exit(1)
    
    def print_welcome(self) -> None:
        """Print welcome message and instructions."""
        print("\n" + "="*60)
        ColorPrinter.print_info("🧠 Neural Small Language Model (SLM) - Interactive CLI")
        print("="*60)
        ColorPrinter.print_info("Commands:")
        ColorPrinter.print_info("  - Type your prompt and press Enter to generate text")
        ColorPrinter.print_info("  - Type 'quit', 'exit', or 'q' to exit")
        ColorPrinter.print_info("  - Type 'help' for generation parameters")
        ColorPrinter.print_info("  - Type 'config' to change settings")
        print("="*60 + "\n")
    
    def print_help(self) -> None:
        """Print help information about generation parameters."""
        ColorPrinter.print_info("Generation Parameters:")
        ColorPrinter.print_info(f"  - Temperature: {getattr(self, 'temperature', 0.8)} (creativity: 0.1-2.0)")
        ColorPrinter.print_info(f"  - Max length: {getattr(self, 'max_length', 100)} tokens")
        ColorPrinter.print_info(f"  - Top-k: {getattr(self, 'top_k', 50)} (diversity)")
        ColorPrinter.print_info(f"  - Top-p: {getattr(self, 'top_p', 0.9)} (nucleus sampling)")
        ColorPrinter.print_info(f"  - Confidence threshold: {self.generator.confidence_threshold:.2f}")
    
    def configure_settings(self) -> None:
        """Allow user to modify generation settings."""
        ColorPrinter.print_info("Current settings:")
        self.print_help()
        
        try:
            print("\nEnter new values (press Enter to keep current):")
            
            # Temperature
            temp_input = input(f"Temperature ({getattr(self, 'temperature', 0.8)}): ").strip()
            if temp_input:
                self.temperature = max(0.1, min(2.0, float(temp_input)))
            elif not hasattr(self, 'temperature'):
                self.temperature = 0.8
            
            # Max length
            length_input = input(f"Max length ({getattr(self, 'max_length', 100)}): ").strip()
            if length_input:
                self.max_length = max(10, min(500, int(length_input)))
            elif not hasattr(self, 'max_length'):
                self.max_length = 100
            
            # Top-k
            topk_input = input(f"Top-k ({getattr(self, 'top_k', 50)}): ").strip()
            if topk_input:
                self.top_k = max(1, min(200, int(topk_input)))
            elif not hasattr(self, 'top_k'):
                self.top_k = 50
            
            # Top-p
            topp_input = input(f"Top-p ({getattr(self, 'top_p', 0.9)}): ").strip()
            if topp_input:
                self.top_p = max(0.1, min(1.0, float(topp_input)))
            elif not hasattr(self, 'top_p'):
                self.top_p = 0.9
            
            # Confidence threshold
            conf_input = input(f"Confidence threshold ({self.generator.confidence_threshold}): ").strip()
            if conf_input:
                self.generator.confidence_threshold = max(0.0, min(1.0, float(conf_input)))
            
            ColorPrinter.print_info("Settings updated successfully!")
            
        except ValueError:
            ColorPrinter.print_error("Invalid input. Settings unchanged.")
        except KeyboardInterrupt:
            ColorPrinter.print_info("Configuration cancelled.")
    
    def run_interactive_loop(self) -> None:
        """Run the main interactive CLI loop."""
        self.print_welcome()
        
        # Initialize default settings
        if not hasattr(self, 'temperature'):
            self.temperature = 0.8
            self.max_length = 100
            self.top_k = 50
            self.top_p = 0.9
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("\n💭 Enter your prompt: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self.print_help()
                    continue
                elif user_input.lower() == 'config':
                    self.configure_settings()
                    continue
                elif not user_input:
                    ColorPrinter.print_info("Please enter a prompt or 'help' for assistance.")
                    continue
                
                # Display user prompt
                ColorPrinter.print_prompt(user_input)
                
                # Generate response
                ColorPrinter.print_info("Generating response...")
                
                generated_text, confidence = self.generator.generate_text(
                    user_input,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p
                )
                
                # Check confidence and provide response
                if self.generator.is_confident(confidence):
                    ColorPrinter.print_answer(generated_text)
                else:
                    ColorPrinter.print_answer(self.generator.fallback_response)
                    ColorPrinter.print_info(f"Low confidence response. Generated: {generated_text}")
                
                # Display confidence
                ColorPrinter.print_confidence(confidence)
        
        except KeyboardInterrupt:
            pass
        finally:
            ColorPrinter.print_info("Thanks for using Neural SLM! Goodbye! 👋")


def main() -> None:
    """Main function to start the interactive CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural SLM Interactive CLI')
    parser.add_argument('--model_path', type=str, default='slm.pkl',
                       help='Path to trained model file')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                       help='Confidence threshold for responses')
    
    args = parser.parse_args()
    
    # Initialize and run interface
    interface = SLMInterface(args.model_path)
    if args.confidence_threshold != 0.3:
        interface.generator.confidence_threshold = args.confidence_threshold
    
    interface.run_interactive_loop()


if __name__ == '__main__':
    main()