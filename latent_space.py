import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from jepa_encoder import JEPAEncoder

class LatentSpaceVisualizer:
    def __init__(self):
        try:
            print("Initializing tokenizer...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            print("Initializing model...", flush=True)
            self.model = JEPAEncoder()
            print("Setting model to eval mode...", flush=True)
            self.model.eval()
            print("Initialization complete!", flush=True)
        except Exception as e:
            print(f"Error during initialization: {str(e)}", flush=True)
            raise

    def create_masked_sequence(self, text):
        """Create sequence with masking positions"""
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask'].to(dtype=torch.float32)
        
        # Create mask for special tokens using tokenizer's methods
        special_tokens_mask = torch.tensor([
            [1 if id in self.tokenizer.all_special_ids else 0 for id in ids]
            for ids in input_ids
        ])
        
        # Identify valid tokens (non-padding, non-special)
        valid_tokens = (attention_mask == 1) & (special_tokens_mask == 0)
        
        # Randomly select 15% of valid tokens to mask
        num_tokens = valid_tokens.sum().item()
        num_masked = max(1, int(0.15 * num_tokens))
        
        # Create masking tensor
        is_masked = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(input_ids.size(0)):
            # Get positions of valid tokens
            valid_positions = valid_tokens[i].nonzero().squeeze(-1)
            # Randomly select positions to mask
            if valid_positions.numel() > 0:
                masked_positions = valid_positions[torch.randperm(valid_positions.numel())[:num_masked]]
                is_masked[i, masked_positions] = True
        
        return input_ids, attention_mask, is_masked

    def get_latent_representation(self, text):
        try:
            print(f"\nProcessing text: {text[:30]}...", flush=True)
            
            # Create sequence with masking positions
            input_ids, attention_mask, is_masked = self.create_masked_sequence(text)
            
            # Print masked and unmasked tokens for visualization
            all_tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0]]
            masked_tokens = [tok if not mask else '[MASK]' for tok, mask in zip(all_tokens, is_masked[0])]
            target_tokens = [tok if mask else '_' for tok, mask in zip(all_tokens, is_masked[0])]
            
            print("Full sequence:", ' '.join(all_tokens), flush=True)
            print("Masked sequence:", ' '.join(masked_tokens), flush=True)
            print("Target tokens:", ' '.join(target_tokens), flush=True)
            
            # Generate embeddings and predictions
            print("Generating embeddings and predictions...", flush=True)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    is_masked_positions=is_masked
                )
            print("Embeddings and predictions generated", flush=True)
            
            # Print prediction metrics
            print("\nPrediction Metrics:", flush=True)
            print(f"Energy: {outputs['energy'][0]:.4f}")
            print(f"Energy Loss: {outputs['energy_loss']:.4f}")
            print(f"Cosine Similarity: {outputs['cosine_similarity'][0]:.4f}")
            print(f"L2 Distance: {outputs['l2_distance'][0]:.4f}")
            
            # Concatenate predicted and actual target embeddings
            joint_embedding = np.concatenate([
                outputs['predicted_target'].numpy(),
                outputs['actual_target'].numpy()
            ], axis=1)
            print(f"Joint embedding shape: {joint_embedding.shape}", flush=True)
            
            # Return embeddings and all metrics
            return joint_embedding, {
                'energy': outputs['energy'][0].item(),
                'energy_loss': outputs['energy_loss'].item(),
                'per_sample_energy': outputs['per_sample_energy'][0].item(),
                'cosine_sim': outputs['cosine_similarity'][0].item(),
                'l2_dist': outputs['l2_distance'][0].item()
            }
            
        except Exception as e:
            print(f"Error during latent representation: {str(e)}", flush=True)
            raise

    def visualize_latent_space(self, texts):
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            print(f"\nProcessing {len(texts)} texts...", flush=True)
            # Get joint embeddings for all texts
            embeddings = []
            metrics = []
            for i, text in enumerate(texts):
                print(f"\nProcessing text {i+1}/{len(texts)}: {text[:30]}...", flush=True)
                embedding, text_metrics = self.get_latent_representation(text)
                print(f"Embedding {i+1} processed successfully", flush=True)
                embeddings.append(embedding)
                metrics.append(text_metrics)
            
            print("\nCreating embeddings array...", flush=True)
            embeddings = np.vstack(embeddings)
            print(f"Embeddings array shape: {embeddings.shape}", flush=True)
            
            # Create visualization using PCA if dimension > 2
            if embeddings.shape[1] > 2:
                print("Applying PCA...", flush=True)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
                print("PCA complete", flush=True)
            else:
                embeddings_2d = embeddings
                
            # Plot the embeddings with energy-focused visualizations
            print("\nCreating energy-based visualizations...", flush=True)
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 20))
            gs = plt.GridSpec(3, 2, figure=fig)
            fig.suptitle('JEPA Energy-Based Analysis', fontsize=16, y=0.95)
            
            # Plot 1 (Large): Energy landscape
            ax1 = fig.add_subplot(gs[0:2, 0])
            scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                c=[m['energy'] for m in metrics],
                                cmap='viridis', alpha=0.8, s=100)
            ax1.set_title('Energy Landscape in Latent Space', fontsize=14)
            ax1.set_xlabel('First Principal Component')
            ax1.set_ylabel('Second Principal Component')
            plt.colorbar(scatter, ax=ax1, label='Energy Score')
            ax1.grid(True)
            
            # Add text annotations for high/low energy points
            energies = [m['energy'] for m in metrics]
            min_idx = np.argmin(energies)
            max_idx = np.argmax(energies)
            ax1.annotate(f'Min Energy: {energies[min_idx]:.3f}',
                        (embeddings_2d[min_idx, 0], embeddings_2d[min_idx, 1]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->'))
            ax1.annotate(f'Max Energy: {energies[max_idx]:.3f}',
                        (embeddings_2d[max_idx, 0], embeddings_2d[max_idx, 1]),
                        xytext=(10, -10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5),
                        arrowprops=dict(arrowstyle='->'))
            
            # Plot 2: 3D Energy Landscape
            ax2 = fig.add_subplot(gs[0:2, 1], projection='3d')
            scatter3d = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], energies,
                                  c=energies, cmap='viridis', alpha=0.8)
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            ax2.set_zlabel('Energy')
            ax2.set_title('3D Energy Landscape')
            plt.colorbar(scatter3d, ax=ax2, label='Energy Score')
            
            # Add surface plot for continuous energy landscape
            if len(embeddings_2d) > 3:  # Need at least 4 points for interpolation
                from scipy.interpolate import griddata
                x = embeddings_2d[:, 0]
                y = embeddings_2d[:, 1]
                z = energies
                
                # Create grid for surface plot
                xi = np.linspace(x.min(), x.max(), 100)
                yi = np.linspace(y.min(), y.max(), 100)
                xi, yi = np.meshgrid(xi, yi)
                
                # Interpolate energy values
                zi = griddata((x, y), z, (xi, yi), method='cubic')
                
                # Plot the surface
                surf = ax2.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.6)
                plt.colorbar(surf, ax=ax2, label='Interpolated Energy')
            
            # Plot 3: Energy Distribution
            ax3 = fig.add_subplot(gs[2, 0])
            ax3.hist(energies, bins=20, density=True, color='skyblue', alpha=0.7)
            ax3.axvline(np.mean(energies), color='red', linestyle='--', label='Mean')
            ax3.axvline(np.median(energies), color='green', linestyle='--', label='Median')
            ax3.set_title('Energy Score Distribution')
            ax3.set_xlabel('Energy')
            ax3.set_ylabel('Density')
            ax3.legend()
            
            # Plot 4: Energy vs. Distance Correlation
            ax4 = fig.add_subplot(gs[2, 1])
            ax4.scatter([m['energy'] for m in metrics],
                       [m['l2_dist'] for m in metrics],
                       alpha=0.6, c=[m['cosine_sim'] for m in metrics],
                       cmap='RdYlBu')
            ax4.set_title('Energy vs. L2 Distance\n(colored by cosine similarity)')
            ax4.set_xlabel('Energy Score')
            ax4.set_ylabel('L2 Distance')
            ax4.grid(True)
            
            # Add summary statistics
            stats_text = (f"Energy Statistics:\n"
                         f"Mean: {np.mean(energies):.3f}\n"
                         f"Median: {np.median(energies):.3f}\n"
                         f"Std: {np.std(energies):.3f}\n"
                         f"Min: {np.min(energies):.3f}\n"
                         f"Max: {np.max(energies):.3f}")
            fig.text(0.02, 0.02, stats_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            print("Saving plots...", flush=True)
            plt.savefig('energy_landscape_analysis.png', bbox_inches='tight', dpi=300)
            plt.close()
            print("\nVisualization saved as 'energy_landscape_analysis.png'", flush=True)
            
            # Print energy statistics
            print("\nEnergy Statistics:")
            print(f"Mean Energy: {np.mean(energies):.4f}")
            print(f"Median Energy: {np.median(energies):.4f}")
            print(f"Std Dev: {np.std(energies):.4f}")
            print(f"Min Energy: {np.min(energies):.4f}")
            print(f"Max Energy: {np.max(energies):.4f}")
            
            return embeddings, metrics
        except Exception as e:
            print(f"\nError during visualization: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            raise

def main():
    try:
        print("Starting visualization process...", flush=True)
        visualizer = LatentSpaceVisualizer()
        
        text = input("\nEnter text to get latent representation: ")
        if text:
            print("\nGenerating latent representation...", flush=True)
            embedding, metrics = visualizer.get_latent_representation(text)
            print("\nLatent representation shape:", embedding.shape)
            print("\nPrediction Metrics:")
            print(f"Energy: {metrics['energy']:.4f}")
            print(f"Energy Loss: {metrics['energy_loss']:.4f}")
            print(f"Cosine Similarity: {metrics['cosine_sim']:.4f}")
            print(f"L2 Distance: {metrics['l2_dist']:.4f}")
            print("\nPredicted vs Actual Target Embeddings:")
            print(embedding)
            
    except Exception as e:
        print(f"Fatal error: {str(e)}", flush=True)
        raise

if __name__ == "__main__":
    main() 