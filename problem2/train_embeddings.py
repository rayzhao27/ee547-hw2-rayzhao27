import re
import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from typing import List, Dict
from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset



def load_arxiv_data(filepath: str) -> List[Dict]:
    try:
        with open(filepath, 'r') as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} papers from {filepath}")
        return papers
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return []


def clean_text(text: str) -> List[str]:
    text = text.lower()
    
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    words = text.split()
    
    words = [word for word in words if len(word) >= 2]
    
    return words


class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding


def determine_architecture(vocab_size: int, max_params: int = 2_000_000):

    best_config = None
    best_params = 0
    
    for hidden_dim in [64, 128, 256, 512]:
        for embedding_dim in [32, 64, 128, 256, 512]:
            # Calculate parameters
            encoder_params = vocab_size * hidden_dim + hidden_dim + hidden_dim * embedding_dim + embedding_dim
            decoder_params = embedding_dim * hidden_dim + hidden_dim + hidden_dim * vocab_size + vocab_size
            total_params = encoder_params + decoder_params
            
            if total_params <= max_params and total_params > best_params:
                best_params = total_params
                best_config = (hidden_dim, embedding_dim)
    
    return best_config, best_params


def train_autoencoder(model, dataloader, num_epochs, output_dir):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training autoencoder...")
    start_time = time.time()
    
    model.train()
    final_loss = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_data, batch_target in dataloader:
            reconstruction, embedding = model(batch_data)
            loss = criterion(reconstruction, batch_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        final_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training complete in {training_time:.1f} seconds")
    
    return model, final_loss, training_time


def main():
    parser = argparse.ArgumentParser(description='Train text embedding autoencoder')
    parser.add_argument('input_papers', help='Input papers JSON file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading abstracts from {args.input_papers}...")
    papers = load_arxiv_data(args.input_papers)
    
    if not papers:
        print("No data loaded. Exiting.")
        return
    
    print(f"Found {len(papers)} abstracts")
    
    abstracts = [paper['abstract'] for paper in papers]
    
    print("Building vocabulary from abstracts...")
    all_words = []
    for abstract in abstracts:
        words = clean_text(abstract)
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    print(f"Building vocabulary from {len(all_words):,} words...")
    print(f"Vocabulary size: {len(word_counts)} words")
    
    vocab_size = min(5000, len(word_counts))
    most_frequent = word_counts.most_common(vocab_size - 1)
    
    word_to_idx = {'<UNK>': 0}
    for idx, (word, count) in enumerate(most_frequent, 1):
        word_to_idx[word] = idx
    
    vocab_size = len(word_to_idx)
    
    config, total_params = determine_architecture(vocab_size)
    hidden_dim, embedding_dim = config
    
    encoder_params = vocab_size * hidden_dim + hidden_dim + hidden_dim * embedding_dim + embedding_dim
    decoder_params = embedding_dim * hidden_dim + hidden_dim + hidden_dim * vocab_size + vocab_size
    total_params = encoder_params + decoder_params
    
    print(f"Model architecture: {vocab_size} → {hidden_dim} → {embedding_dim} → {hidden_dim} → {vocab_size}")
    print(f"Total parameters: {total_params:,} (under 2,000,000 limit ✓)")
    
    print("Creating bag-of-words representations...")
    bow_vectors = []
    
    for abstract in abstracts:
        words = clean_text(abstract)
        bow = [0.0] * vocab_size
        
        for word in words:
            idx = word_to_idx.get(word, 0)
            bow[idx] += 1.0
        
        total_words = sum(bow)
        if total_words > 0:
            bow = [count / total_words for count in bow]
        
        bow_vectors.append(bow)
    
    data = torch.tensor(bow_vectors, dtype=torch.float32)
    
    dataset = TensorDataset(data, data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = TextAutoencoder(vocab_size, hidden_dim, embedding_dim)
    
    trained_model, final_loss, training_time = train_autoencoder(model, dataloader, args.epochs, args.output_dir)
    
    print("Generating embeddings for all papers...")
    trained_model.eval()
    embeddings_data = []
    
    with torch.no_grad():
        for i, (bow_vector, _) in enumerate(dataset):
            bow_input = bow_vector.unsqueeze(0)
            reconstruction, embedding = trained_model(bow_input)
            
            criterion = nn.BCELoss()
            recon_loss = criterion(reconstruction, bow_input).item()
            
            embeddings_data.append({
                "arxiv_id": papers[i]['arxiv_id'],
                "embedding": embedding.squeeze(0).tolist(),
                "reconstruction_loss": recon_loss
            })
    
    model_path = os.path.join(args.output_dir, "model.pth")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'vocab_to_idx': word_to_idx,
        'model_config': {
            'vocab_size': vocab_size,
            'hidden_dim': hidden_dim,
            'embedding_dim': embedding_dim
        }
    }, model_path)
    
    embeddings_path = os.path.join(args.output_dir, "embeddings.json")
    with open(embeddings_path, 'w') as f:
        json.dump(embeddings_data, f, indent=2)
    
    vocab_path = os.path.join(args.output_dir, "vocabulary.json")
    idx_to_vocab = {str(idx): word for word, idx in word_to_idx.items()}
    vocab_data = {
        "vocab_to_idx": word_to_idx,
        "idx_to_vocab": idx_to_vocab,
        "vocab_size": vocab_size,
        "total_words": len(all_words)
    }
    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    from datetime import datetime, timezone
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(seconds=training_time)
    
    training_log_path = os.path.join(args.output_dir, "training_log.json")
    training_log = {
        "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "epochs": args.epochs,
        "final_loss": final_loss,
        "total_parameters": total_params,
        "papers_processed": len(papers),
        "embedding_dimension": embedding_dim
    }
    with open(training_log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\n-Training completed successfully!")
    print(f"Output files saved to {args.output_dir}:")
    print(f"  - model.pth: Trained PyTorch model")
    print(f"  - embeddings.json: Generated embeddings for all papers")
    print(f"  - vocabulary.json: Vocabulary mapping")
    print(f"  - training_log.json: Training metadata")


if __name__ == "__main__":
    main()
