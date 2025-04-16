import torch
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import os
from torch.serialization import safe_globals
import numpy as np
import pandas as pd

def load_embeddings():
    # Load the embeddings from local file
    embeddings_path = "./model/word_embeddings_dim100_epoch10.pt"
    print(f"Loading embeddings from {embeddings_path}...")
    
    # Use safe_globals context manager to allow numpy._core.multiarray._reconstruct
    with safe_globals([np._core.multiarray._reconstruct]):
        embedding_dict = torch.load(embeddings_path, weights_only=False)
    
    return embedding_dict

def prepare_data(embedding_dict, num_words=None):
    # Convert embeddings to numpy array and get corresponding words
    words = list(embedding_dict.keys())
    if num_words is not None:
        words = words[:num_words]
    
    embeddings = np.array([embedding_dict[word] for word in words])
    return words, embeddings

def visualize_embeddings(words, embeddings, output_path="./embeddings_visualization.png", num_labels=100):
    # Reduce dimensionality to 2D using UMAP
    print("Reducing dimensionality with UMAP...")
    umap = UMAP(n_components=2, random_state=42)
    embeddings_2d = umap.fit_transform(embeddings)
    
    # Create the visualization
    plt.figure(figsize=(20, 20))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
    
    # Add labels for some words (to avoid overcrowding)
    num_labels = min(num_labels, len(words))
    for i in range(num_labels):
        plt.annotate(words[i], 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=6)
    
    plt.title("Word Embeddings Visualization (UMAP)")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Also create an interactive plot using plotly
    try:
        import plotly.express as px
        import plotly.io as pio
        
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'word': words
        })
        
        fig = px.scatter(df, x='x', y='y', text='word', title='Word Embeddings Visualization (UMAP)')
        fig.update_traces(textposition='top center', marker=dict(size=5))
        
        # Save interactive HTML
        pio.write_html(fig, './embeddings_visualization.html')
        print("Interactive visualization saved to embeddings_visualization.html")
    except ImportError:
        print("Plotly not installed. Skipping interactive visualization.")

def main():
    # Load embeddings
    embedding_dict = load_embeddings()
    
    # Prepare data (using all words)
    words, embeddings = prepare_data(embedding_dict)
    
    # Visualize embeddings
    visualize_embeddings(words, embeddings, num_labels=200)
    
    # Print some statistics
    print(f"\nTotal vocabulary size: {len(embedding_dict)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Words visualized: {len(words)}")

if __name__ == "__main__":
    main() 