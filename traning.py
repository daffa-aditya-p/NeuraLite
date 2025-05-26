#!/usr/bin/env python3
"""
Neural Small Language Model (SLM) Training Script
A lightweight transformer-based language model with custom backpropagation.
"""

import argparse
import glob
import pickle
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


class ColorLogger:
    """Colored logging utility for training progress."""
    
    COLORS = {
        'INFO': '\033[94m',     # Blue
        'EPOCH': '\033[93m',   # Yellow
        'LOSS': '\033[91m',    # Red
        'SUCCESS': '\033[92m', # Green
        'RESET': '\033[0m'     # Reset
    }
    
    @classmethod
    def log(cls, level: str, message: str) -> None:
        """Log colored messages with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        color = cls.COLORS.get(level, cls.COLORS['RESET'])
        print(f"{color}[{level}] {timestamp}: {message}{cls.COLORS['RESET']}")


class ProgressBar:
    """Simple progress bar for training visualization."""
    
    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width
        self.current = 0
    
    def update(self, increment: int = 1) -> None:
        """Update progress bar."""
        self.current += increment
        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = '█' * filled + '░' * (self.width - filled)
        percentage = progress * 100
        print(f'\r[{bar}] {percentage:.1f}% ({self.current}/{self.total})', end='', flush=True)
    
    def finish(self) -> None:
        """Complete the progress bar."""
        print()


class Tokenizer:
    """Character or subword tokenizer for text preprocessing."""
    
    def __init__(self, vocab_size: int = 5000, mode: str = 'char'):
        self.vocab_size = vocab_size
        self.mode = mode
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from training texts."""
        ColorLogger.log('INFO', f'Building vocabulary in {self.mode} mode...')
        
        if self.mode == 'char':
            # Character-level tokenization
            all_chars = set()
            for text in texts:
                all_chars.update(text)
            
            # Create vocab with special tokens first
            self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
            
            # Add characters
            for i, char in enumerate(sorted(all_chars), len(self.special_tokens)):
                if i < self.vocab_size:
                    self.vocab[char] = i
                else:
                    break
        
        else:  # subword mode (simplified BPE-like)
            # Count character pairs and merge most frequent
            word_counts = Counter()
            for text in texts:
                words = text.split()
                for word in words:
                    word_counts[' '.join(word) + ' </w>'] += 1
            
            # Start with character vocab
            vocab_set = set()
            for word in word_counts:
                vocab_set.update(word.split())
            
            # Simple merging process
            for _ in range(self.vocab_size // 4):  # Limited merges
                pairs = Counter()
                for word, count in word_counts.items():
                    symbols = word.split()
                    for i in range(len(symbols) - 1):
                        pairs[(symbols[i], symbols[i + 1])] += count
                
                if not pairs:
                    break
                
                best_pair = pairs.most_common(1)[0][0]
                new_word_counts = {}
                pattern = re.escape(' '.join(best_pair))
                replacement = ''.join(best_pair)
                
                for word in word_counts:
                    new_word = re.sub(pattern, replacement, word)
                    new_word_counts[new_word] = word_counts[word]
                
                word_counts = new_word_counts
                vocab_set.add(replacement)
            
            # Create final vocab
            self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
            for i, token in enumerate(sorted(vocab_set), len(self.special_tokens)):
                if i < self.vocab_size:
                    self.vocab[token] = i
                else:
                    break
        
        self.inv_vocab = {i: token for token, i in self.vocab.items()}
        ColorLogger.log('SUCCESS', f'Vocabulary built with {len(self.vocab)} tokens')
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.mode == 'char':
            tokens = [self.vocab.get(char, self.vocab['<UNK>']) for char in text]
        else:
            # Simplified subword encoding
            tokens = []
            words = text.split()
            for word in words:
                # Try to find subword matches
                i = 0
                while i < len(word):
                    found = False
                    for length in range(min(len(word) - i, 10), 0, -1):
                        subword = word[i:i + length]
                        if subword in self.vocab:
                            tokens.append(self.vocab[subword])
                            i += length
                            found = True
                            break
                    if not found:
                        tokens.append(self.vocab['<UNK>'])
                        i += 1
        
        return [self.vocab['<BOS>']] + tokens + [self.vocab['<EOS>']]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.inv_vocab.get(token_id, '<UNK>') for token_id in token_ids]
        text = ''.join(tokens) if self.mode == 'char' else ' '.join(tokens)
        # Clean special tokens
        for special in self.special_tokens:
            text = text.replace(special, '')
        return text.strip()


class TransformerBlock:
    """Single transformer decoder block with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Multi-head attention parameters
        self.wq = np.random.normal(0, 0.02, (d_model, d_model))
        self.wk = np.random.normal(0, 0.02, (d_model, d_model))
        self.wv = np.random.normal(0, 0.02, (d_model, d_model))
        self.wo = np.random.normal(0, 0.02, (d_model, d_model))
        
        # Feed-forward parameters
        self.w1 = np.random.normal(0, 0.02, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.normal(0, 0.02, (d_ff, d_model))
        self.b2 = np.zeros(d_model)
        
        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-8) + beta
    
    def masked_self_attention(self, x: np.ndarray) -> np.ndarray:
        """Masked multi-head self-attention."""
        batch_size, seq_len, d_model = x.shape
        head_dim = d_model // self.n_heads
        
        # Compute Q, K, V
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        
        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        
        # Apply causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask
        
        # Softmax
        attention_weights = self.softmax(scores)
        
        # Apply attention
        out = attention_weights @ v
        
        # Concatenate heads
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Output projection
        return out @ self.wo
    
    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward network with ReLU activation."""
        return self.relu(x @ self.w1 + self.b1) @ self.w2 + self.b2
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through transformer block."""
        # Self-attention with residual connection and layer norm
        attn_out = self.masked_self_attention(x)
        x = self.layer_norm(x + attn_out, self.ln1_gamma, self.ln1_beta)
        
        # Feed-forward with residual connection and layer norm
        ff_out = self.feed_forward(x)
        x = self.layer_norm(x + ff_out, self.ln2_gamma, self.ln2_beta)
        
        return x
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)


class NeuralSLM:
    """Neural Small Language Model with transformer architecture."""
    
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8, 
                 n_layers: int = 4, d_ff: int = 1024, max_seq_len: int = 512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embeddings = np.random.normal(0, 0.02, (vocab_size, d_model))
        
        # Positional encodings (sinusoidal)
        self.pos_encodings = self._create_positional_encodings()
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        
        # Output layer
        self.output_layer = np.random.normal(0, 0.02, (d_model, vocab_size))
        self.output_bias = np.zeros(vocab_size)
        
        # Layer norm for final output
        self.final_ln_gamma = np.ones(d_model)
        self.final_ln_beta = np.zeros(d_model)
    
    def _create_positional_encodings(self) -> np.ndarray:
        """Create sinusoidal positional encodings."""
        pos_enc = np.zeros((self.max_seq_len, self.d_model))
        
        for pos in range(self.max_seq_len):
            for i in range(0, self.d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / self.d_model)))
        
        return pos_enc
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass through the model."""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embeddings[input_ids]
        
        # Add positional encodings
        x += self.pos_encodings[:seq_len]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Final layer normalization
        x = self._layer_norm(x, self.final_ln_gamma, self.final_ln_beta)
        
        # Output projection
        logits = x @ self.output_layer + self.output_bias
        
        return logits
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-8) + beta


class AdamOptimizer:
    """Adam optimizer implementation."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """Update parameters using Adam optimization."""
        self.t += 1
        
        for name, param in params.items():
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            
            grad = grads[name]
            
            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


def load_datasets(data_path: str = "*.txt") -> List[str]:
    """Load and merge all text datasets."""
    ColorLogger.log('INFO', f'Loading datasets from {data_path}...')
    
    files = glob.glob(data_path)
    if not files:
        ColorLogger.log('LOSS', f'No files found matching {data_path}')
        return []
    
    texts = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append(content)
            ColorLogger.log('INFO', f'Loaded {file_path} ({len(content)} chars)')
        except Exception as e:
            ColorLogger.log('LOSS', f'Error loading {file_path}: {e}')
    
    ColorLogger.log('SUCCESS', f'Loaded {len(texts)} files')
    return texts


def preprocess_text(texts: List[str]) -> List[str]:
    """Clean and preprocess text data."""
    ColorLogger.log('INFO', 'Preprocessing texts...')
    
    processed = []
    for text in texts:
        # Lowercase
        text = text.lower()
        
        # Clean with regex (keep letters, numbers, basic punctuation, spaces)
        text = re.sub(r'[^a-z0-9\s\.,!?;:\-\'\"]+', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        processed.append(text.strip())
    
    ColorLogger.log('SUCCESS', 'Text preprocessing completed')
    return processed


def create_training_data(texts: List[str], tokenizer: Tokenizer, 
                        seq_len: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Create training sequences from tokenized texts."""
    ColorLogger.log('INFO', f'Creating training sequences (seq_len={seq_len})...')
    
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(0, len(all_tokens) - seq_len, seq_len // 2):  # Overlapping sequences
        seq = all_tokens[i:i + seq_len]
        target = all_tokens[i + 1:i + seq_len + 1]
        
        if len(seq) == seq_len and len(target) == seq_len:
            sequences.append(seq)
            targets.append(target)
    
    ColorLogger.log('SUCCESS', f'Created {len(sequences)} training sequences')
    return np.array(sequences), np.array(targets)


def compute_loss_and_gradients(model: NeuralSLM, input_ids: np.ndarray, 
                              targets: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
    """Compute cross-entropy loss and gradients (simplified backprop)."""
    batch_size, seq_len = input_ids.shape
    
    # Forward pass
    logits = model.forward(input_ids)
    
    # Compute cross-entropy loss
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Cross-entropy loss
    loss = 0.0
    for b in range(batch_size):
        for s in range(seq_len):
            target_token = targets[b, s]
            if target_token < model.vocab_size:  # Valid token
                loss -= np.log(probs[b, s, target_token] + 1e-8)
    
    loss /= (batch_size * seq_len)
    
    # Simplified gradient computation (approximate)
    grads = {}
    
    # Output layer gradients
    d_logits = probs.copy()
    for b in range(batch_size):
        for s in range(seq_len):
            target_token = targets[b, s]
            if target_token < model.vocab_size:
                d_logits[b, s, target_token] -= 1.0
    
    d_logits /= (batch_size * seq_len)
    
    # Gradient for output layer
    # Get final layer features (simplified)
    final_features = model.token_embeddings[input_ids]  # Simplified
    grads['output_layer'] = np.mean(final_features, axis=(0, 1)).reshape(-1, 1) @ np.mean(d_logits, axis=(0, 1)).reshape(1, -1)
    grads['output_bias'] = np.mean(d_logits, axis=(0, 1))
    
    # Simplified embedding gradients
    d_embeddings = np.zeros_like(model.token_embeddings)
    for b in range(batch_size):
        for s in range(seq_len):
            token_id = input_ids[b, s]
            if token_id < model.vocab_size:
                d_embeddings[token_id] += np.mean(d_logits[b, s]) * 0.01  # Simplified
    
    grads['token_embeddings'] = d_embeddings
    
    return loss, grads


def train_model(args: argparse.Namespace) -> None:
    """Main training function."""
    ColorLogger.log('INFO', 'Starting Neural SLM training...')
    
    # Load and preprocess data
    texts = load_datasets(args.data_path)
    if not texts:
        ColorLogger.log('LOSS', 'No training data found. Exiting.')
        return
    
    texts = preprocess_text(texts)
    
    # Initialize tokenizer and build vocabulary
    tokenizer = Tokenizer(vocab_size=args.vocab_size, mode=args.tokenizer_mode)
    tokenizer.build_vocab(texts)
    
    # Create training data
    train_inputs, train_targets = create_training_data(texts, tokenizer, args.seq_len)
    
    if len(train_inputs) == 0:
        ColorLogger.log('LOSS', 'No training sequences created. Exiting.')
        return
    
    # Initialize model
    model = NeuralSLM(
        vocab_size=len(tokenizer.vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len
    )
    
    # Initialize optimizer
    optimizer = AdamOptimizer(learning_rate=args.learning_rate)
    
    # Training loop
    ColorLogger.log('INFO', f'Training for {args.epochs} epochs...')
    
    for epoch in range(args.epochs):
        ColorLogger.log('EPOCH', f'Epoch {epoch + 1}/{args.epochs}')
        
        # Shuffle training data
        indices = np.random.permutation(len(train_inputs))
        train_inputs = train_inputs[indices]
        train_targets = train_targets[indices]
        
        epoch_loss = 0.0
        num_batches = len(train_inputs) // args.batch_size
        
        progress_bar = ProgressBar(num_batches)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + args.batch_size
            
            batch_inputs = train_inputs[start_idx:end_idx]
            batch_targets = train_targets[start_idx:end_idx]
            
            # Compute loss and gradients
            loss, grads = compute_loss_and_gradients(model, batch_inputs, batch_targets)
            epoch_loss += loss
            
            # Update parameters
            params = {
                'token_embeddings': model.token_embeddings,
                'output_layer': model.output_layer,
                'output_bias': model.output_bias
            }
            
            optimizer.update(params, grads)
            
            progress_bar.update()
        
        progress_bar.finish()
        
        avg_loss = epoch_loss / num_batches
        ColorLogger.log('LOSS', f'Average loss: {avg_loss:.4f}')
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            save_model(model, tokenizer, f'slm_checkpoint_epoch_{epoch + 1}.pkl')
    
    # Save final model
    save_model(model, tokenizer, args.output_path)
    ColorLogger.log('SUCCESS', f'Training completed! Model saved to {args.output_path}')


def save_model(model: NeuralSLM, tokenizer: Tokenizer, path: str) -> None:
    """Save model and tokenizer to pickle file."""
    try:
        model_data = {
            'model': model,
            'tokenizer': tokenizer,
            'vocab_size': len(tokenizer.vocab),
            'config': {
                'd_model': model.d_model,
                'n_heads': model.n_heads,
                'n_layers': model.n_layers,
                'd_ff': model.d_ff,
                'max_seq_len': model.max_seq_len
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        ColorLogger.log('SUCCESS', f'Model saved to {path}')
    except Exception as e:
        ColorLogger.log('LOSS', f'Error saving model: {e}')


def main() -> None:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train Neural Small Language Model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='*.txt',
                       help='Path pattern for training text files')
    parser.add_argument('--output_path', type=str, default='slm.pkl',
                       help='Output path for trained model')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=5000,
                       help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1024,
                       help='Feed-forward dimension')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='Sequence length')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    # Tokenizer parameters
    parser.add_argument('--tokenizer_mode', type=str, default='char',
                       choices=['char', 'subword'],
                       help='Tokenizer mode: char or subword')
    
    args = parser.parse_args()
    
    # Display configuration
    ColorLogger.log('INFO', 'Training Configuration:')
    for arg, value in vars(args).items():
        ColorLogger.log('INFO', f'  {arg}: {value}')
    
    try:
        train_model(args)
    except KeyboardInterrupt:
        ColorLogger.log('LOSS', 'Training interrupted by user')
        sys.exit(1)
    except Exception as e:
        ColorLogger.log('LOSS', f'Training failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()