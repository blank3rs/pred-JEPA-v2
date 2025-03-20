import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class JEPAEncoder(nn.Module):
    def __init__(self, hidden_size=768, projection_dim=256, energy_margin=1.0, init_temperature=0.1, momentum=0.999):
        super().__init__()
        print("Initializing shared encoder...", flush=True)
        
        # Initialize configuration
        self.config = AutoConfig.from_pretrained('roberta-base')
        
        # Initialize encoder for context branch
        self.encoder = AutoModel.from_pretrained('roberta-base', local_files_only=False)
        
        # Initialize target encoder as a momentum-updated copy
        self.target_encoder = AutoModel.from_pretrained('roberta-base', local_files_only=False)
        # Disable gradient computation for target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Initialize projection layers for both branches
        print("Initializing projection layers...", flush=True)
        # Context branch projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Target branch projection (momentum-updated copy)
        self.target_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        # Copy initial weights from context projection
        self._copy_weights(self.projection, self.target_projection)
        # Disable gradient computation for target projection
        for param in self.target_projection.parameters():
            param.requires_grad = False
        
        # Momentum update parameter
        self.momentum = momentum
        
        # Initialize learnable temperature parameter
        print("Initializing learnable temperature parameter...", flush=True)
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        
        # Initialize learnable mask embedding
        print("Initializing learnable mask embedding...", flush=True)
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, hidden_size))  # [1, 1, hidden_size] for batch compatibility
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)  # Initialize similar to BERT
        
        # Initialize predictor network
        print("Initializing predictor network...", flush=True)
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # Initialize weights for all layers
        for module in [self.projection, self.predictor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        # Energy function parameters
        self.energy_margin = energy_margin
        
        # Move model to available device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}", flush=True)
        self.to(self.device)
    
    def _apply_embedding_mask(self, token_embeddings, attention_mask, is_masked_positions):
        """Replace embeddings at masked positions with learnable mask embedding"""
        # Create mask for valid positions (excluding padding)
        valid_mask = attention_mask.bool()
        
        # Create mask for positions to replace
        mask_positions = is_masked_positions & valid_mask
        
        # Expand mask embedding to match batch size and sequence length
        batch_size, seq_len, _ = token_embeddings.size()
        expanded_mask_embedding = self.mask_embedding.expand(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        
        # Create masked embeddings tensor
        masked_embeddings = token_embeddings.clone()
        
        # Apply mask embedding using broadcasting
        mask_positions = mask_positions.unsqueeze(-1)  # [batch_size, seq_len, 1]
        masked_embeddings = torch.where(mask_positions, expanded_mask_embedding, masked_embeddings)
        
        return masked_embeddings
    
    def _get_token_embeddings(self, encoder_outputs, attention_mask, positions_mask):
        """Extract embeddings for specified positions"""
        token_embeddings = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Create mask for valid positions (excluding padding and using positions mask)
        valid_mask = attention_mask.bool() & positions_mask
        valid_mask = valid_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Apply masking
        masked_embeddings = token_embeddings * valid_mask
        
        # Average over valid positions
        sum_embeddings = masked_embeddings.sum(dim=1)  # [batch_size, hidden_size]
        num_valid = valid_mask.sum(dim=1)  # [batch_size, 1]
        mean_embeddings = sum_embeddings / torch.clamp(num_valid, min=1e-9)
        
        return mean_embeddings
    
    def train(self, mode=True):
        """Override train mode to properly handle encoder training state"""
        super().train(mode)
        if mode:
            # When in training mode, ensure encoder and mask embedding can receive gradients
            self.encoder.train()
            self.mask_embedding.requires_grad = True
        return self
    
    def eval(self):
        """Override eval mode to properly handle encoder evaluation state"""
        super().eval()
        # Put encoder in eval mode during evaluation
        self.encoder.eval()
        # Freeze mask embedding during evaluation to prevent inference-time drift
        self.mask_embedding.requires_grad = False
        return self

    @torch.no_grad()
    def _copy_weights(self, source_module, target_module):
        """Helper to copy weights between modules"""
        # Create parameter mapping using names
        source_params = dict(source_module.named_parameters())
        target_params = dict(target_module.named_parameters())
        
        # Verify parameter names match
        if set(source_params.keys()) != set(target_params.keys()):
            raise ValueError(
                "Source and target modules have different parameter names. "
                f"Source: {set(source_params.keys())}, Target: {set(target_params.keys())}"
            )
        
        # Copy parameters by name
        for name, source_param in source_params.items():
            target_params[name].data.copy_(source_param.data)
    
    @torch.no_grad()
    def _momentum_update(self):
        """Update target encoder and projection using momentum update with safe parameter matching"""
        # Update encoder parameters using named parameters
        source_params = dict(self.encoder.named_parameters())
        target_params = dict(self.target_encoder.named_parameters())
        
        # Verify encoder parameter names match
        if set(source_params.keys()) != set(target_params.keys()):
            raise ValueError(
                "Encoder and target encoder have different parameter names. "
                f"Encoder: {set(source_params.keys())}, Target: {set(target_params.keys())}"
            )
        
        # Update encoder parameters by name
        for name, source_param in source_params.items():
            target_param = target_params[name]
            target_param.data = target_param.data * self.momentum + source_param.data * (1. - self.momentum)
        
        # Update projection parameters using named parameters
        source_proj_params = dict(self.projection.named_parameters())
        target_proj_params = dict(self.target_projection.named_parameters())
        
        # Verify projection parameter names match
        if set(source_proj_params.keys()) != set(target_proj_params.keys()):
            raise ValueError(
                "Projection and target projection have different parameter names. "
                f"Projection: {set(source_proj_params.keys())}, Target: {set(target_proj_params.keys())}"
            )
        
        # Update projection parameters by name
        for name, source_param in source_proj_params.items():
            target_param = target_proj_params[name]
            target_param.data = target_param.data * self.momentum + source_param.data * (1. - self.momentum)
    
    def encode_sequence(self, input_ids, attention_mask, is_masked_positions, for_target=False):
        """Encode sequence with embedding-level masking"""
        try:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(dtype=torch.float32, device=self.device)
            is_masked_positions = is_masked_positions.to(self.device)
            
            # Create inputs for full embedding computation
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': torch.zeros_like(input_ids) if hasattr(self.encoder.embeddings, 'token_type_embeddings') else None,
                'position_ids': torch.arange(0, input_ids.size(1), device=self.device).expand_as(input_ids)
            }
            
            if for_target:
                # Use target encoder and projection for target branch (no gradients)
                with torch.no_grad():
                    embeddings = self.target_encoder.embeddings(
                        input_ids=inputs['input_ids'],
                        position_ids=inputs['position_ids'],
                        token_type_ids=inputs['token_type_ids']
                    )
                    
                    outputs = self.target_encoder.encoder(
                        embeddings,
                        attention_mask=attention_mask
                    )
                    
                    # Get embeddings for masked positions
                    positions_mask = is_masked_positions
                    embeddings = self._get_token_embeddings(outputs, attention_mask, positions_mask)
                    
                    # Use target projection
                    embeddings = self.target_projection(embeddings)
            else:
                # Use main encoder and projection for context branch (with gradients)
                embeddings = self.encoder.embeddings(
                    input_ids=inputs['input_ids'],
                    position_ids=inputs['position_ids'],
                    token_type_ids=inputs['token_type_ids']
                )
                
                # Apply mask embedding for context view
                embeddings = self._apply_embedding_mask(embeddings, attention_mask, is_masked_positions)
                
                outputs = self.encoder.encoder(
                    embeddings,
                    attention_mask=attention_mask
                )
                
                # Get embeddings for unmasked positions
                positions_mask = ~is_masked_positions
                embeddings = self._get_token_embeddings(outputs, attention_mask, positions_mask)
                
                # Use context projection
                embeddings = self.projection(embeddings)
            
            return embeddings
            
        except Exception as e:
            print(f"Error in sequence encoding: {str(e)}", flush=True)
            raise

    def predict_target(self, context_embedding):
        """Predict target embedding from context embedding"""
        return self.predictor(context_embedding)
    
    def _generate_negative_samples(self, actual_target, num_negatives=None):
        """Generate negative samples using in-batch negatives with queue-based sampling"""
        batch_size = actual_target.size(0)
        
        # If batch size is 1, we need to use historical embeddings or return None
        if batch_size == 1:
            return None
            
        # Default to using all other samples in batch as negatives
        if num_negatives is None:
            num_negatives = batch_size - 1
            
        # Create negative samples by using other targets in the batch
        # For each sample, use all other samples as negatives
        negative_targets = []
        for i in range(batch_size):
            # Get all samples except the current one
            other_samples = torch.cat([actual_target[:i], actual_target[i+1:]], dim=0)
            # If we need more negatives than available, repeat the samples
            if num_negatives > other_samples.size(0):
                other_samples = other_samples.repeat(num_negatives // other_samples.size(0) + 1, 1)
            # Select the required number of negatives
            negative_targets.append(other_samples[:num_negatives])
        
        # Stack all negative samples [batch_size, num_negatives, embedding_dim]
        negative_targets = torch.stack(negative_targets)
        return negative_targets

    def compute_energy(self, predicted_target, actual_target, negative_targets=None):
        """Compute energy-based loss following JEPA's formulation with proper negative sampling"""
        def compute_similarity(x, y):
            """Compute cosine similarity between embeddings"""
            # Normalize embeddings to unit sphere
            x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
            y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)
            # Compute cosine similarity
            return torch.sum(x_norm * y_norm, dim=-1)
        
        # Compute positive pair similarity (in [-1, 1])
        pos_sim = compute_similarity(predicted_target, actual_target)
        # Convert similarity to normalized energy (in [0, 1])
        pos_energy = 0.5 * (1.0 - pos_sim)  # Maps [-1, 1] to [0, 1]
        
        # If no negative_targets provided and batch size > 1, generate them using in-batch negatives
        batch_size = predicted_target.size(0)
        if negative_targets is None and batch_size > 1:
            negative_targets = self._generate_negative_samples(actual_target)
        
        # Handle different batch size scenarios
        if batch_size == 1 or negative_targets is None:
            # For single samples, use direct energy minimization with stability
            energy_loss = pos_energy  # Directly minimize positive pair energy
            
            # Add L2 regularization to prevent collapse
            pred_norm = torch.nn.functional.normalize(predicted_target, p=2, dim=-1)
            target_norm = torch.nn.functional.normalize(actual_target, p=2, dim=-1)
            reg_loss = torch.mean((pred_norm - target_norm) ** 2)
            
            # Simple weighted combination
            total_loss = energy_loss + 0.1 * reg_loss
            
            # Ensure no nan values
            if torch.isnan(total_loss).any():
                print(f"Warning: NaN detected in loss computation")
                print(f"pos_energy: {pos_energy}")
                print(f"reg_loss: {reg_loss}")
                # Fallback to basic L2 loss if energy computation fails
                total_loss = reg_loss
            
            return {
                'energy': pos_energy,
                'loss': total_loss.mean(),
                'per_sample_loss': total_loss,
                'target_energy': self.energy_margin,
                'temperature': self.temperature.detach()
            }
        
        # Multi-sample case remains unchanged
        # Compute negative pair similarities: [batch_size, num_negatives]
        pred_expanded = predicted_target.unsqueeze(1)  # [batch_size, 1, dim]
        neg_sim = compute_similarity(pred_expanded, negative_targets)  # [batch_size, num_negatives]
        # Convert similarities to normalized energies
        neg_energy = 0.5 * (1.0 - neg_sim)  # Maps [-1, 1] to [0, 1]
        
        # Compute InfoNCE-style loss with temperature scaling
        temperature = torch.clamp(self.temperature, min=0.001, max=0.5)
        
        # Scale energies by temperature (lower energy = better match)
        pos_logits = -pos_energy / temperature
        neg_logits = -neg_energy / temperature
        
        # Standard InfoNCE formulation with stability
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        logits = logits - max_logits
        exp_logits = torch.exp(logits)
        
        # Compute loss with numerical stability
        log_denominator = torch.log(exp_logits.sum(dim=1) + 1e-6)
        infonce_loss = -pos_logits + log_denominator
        
        # Add margin term
        margin_loss = torch.clamp(pos_energy - self.energy_margin, min=0.0)
        
        # Combine losses
        total_loss = infonce_loss + 0.1 * margin_loss
        
        return {
            'energy': pos_energy,
            'loss': total_loss.mean(),
            'per_sample_loss': total_loss,
            'margin': self.energy_margin,
            'temperature': self.temperature.detach(),
            'infonce_loss': infonce_loss.mean(),
            'margin_loss': margin_loss.mean()
        }
    
    def compute_prediction_metrics(self, predicted_target, actual_target):
        """Compute detailed prediction metrics"""
        # Compute energy-based metrics
        energy_metrics = self.compute_energy(predicted_target, actual_target)
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(predicted_target, actual_target)
        
        # Compute L2 distance in raw embedding space
        l2_dist = torch.norm(predicted_target - actual_target, dim=1)
        
        return {
            'energy': energy_metrics['energy'],
            'energy_loss': energy_metrics['loss'],
            'per_sample_energy': energy_metrics['per_sample_loss'],
            'cosine_similarity': cos_sim,
            'l2_distance': l2_dist
        }
    
    def forward(self, input_ids, attention_mask, is_masked_positions):
        try:
            # Update target encoder with momentum before forward pass
            if self.training:
                self._momentum_update()
            
            # Get context embedding using unmasked positions (with gradients)
            context_embedding = self.encode_sequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                is_masked_positions=is_masked_positions,
                for_target=False
            )
            
            # Predict target embedding from context
            predicted_target = self.predict_target(context_embedding)
            
            # Get actual target embedding using masked positions (without gradients)
            actual_target = self.encode_sequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                is_masked_positions=is_masked_positions,
                for_target=True  # Uses target encoder with no gradients
            )
            
            # Compute prediction metrics
            metrics = self.compute_prediction_metrics(predicted_target, actual_target)
            
            # Return everything needed for both training and analysis
            return {
                'context_embedding': context_embedding.cpu(),
                'predicted_target': predicted_target.cpu(),
                'actual_target': actual_target.cpu(),
                'energy': metrics['energy'].cpu(),
                'energy_loss': metrics['energy_loss'].cpu(),
                'per_sample_energy': metrics['per_sample_energy'].cpu(),
                'cosine_similarity': metrics['cosine_similarity'].cpu(),
                'l2_distance': metrics['l2_distance'].cpu()
            }
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}", flush=True)
            raise 