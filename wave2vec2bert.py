
import torch
import torch.nn as nn
from .classification_head import ClassificationHead, SpeechClassifierOutput
from transformers import Wav2Vec2BertModel
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, TripletMarginLoss,TripletMarginWithDistanceLoss
from transformers import AutoFeatureExtractor


################################################################################
# 1. MNR Loss
################################################################################
class MNRLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss for multiple negatives per anchor.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_emb, pos_emb, neg_embs):
        """
        anchor_emb: (batch_size, embed_dim)
        pos_emb: (batch_size, embed_dim)
        neg_embs: (batch_size * num_negatives, embed_dim)
        """
        batch_size = anchor_emb.size(0)
        num_negatives = neg_embs.size(0) // batch_size

        # (Optional) normalize if you want unit vectors
        # anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
        # pos_emb    = F.normalize(pos_emb, p=2, dim=1)
        # neg_embs   = F.normalize(neg_embs, p=2, dim=1)

        # 1) Compute cosine similarities
        s_ap = F.cosine_similarity(anchor_emb, pos_emb, dim=1) / self.temperature
        s_an = F.cosine_similarity(
            anchor_emb.unsqueeze(1).repeat(1, num_negatives, 1).view(-1, anchor_emb.size(1)),
            neg_embs,
            dim=1
        ).view(batch_size, num_negatives) / self.temperature

        # 2) Create logits of shape (batch_size, 1 + num_negatives)
        logits = torch.cat([s_ap.unsqueeze(1), s_an], dim=1)

        # 3) The correct class is always 0 (anchor-positive)
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # 4) Cross-entropy
        loss = F.cross_entropy(logits, targets)
        return loss

################################################################################

class Wav2Vec2BERT(nn.Module):
    def __init__(self, model_name, pooling_mode='mean'):
        super().__init__()
        self.num_labels = 2
        self.pooling_mode = pooling_mode
        self.wav2vec2bert = Wav2Vec2BertModel.from_pretrained(model_name)
        self.config = self.wav2vec2bert.config
        self.classifier = ClassificationHead(self.wav2vec2bert.config)

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']"
            )
        return outputs

    def forward(self, 
                input_features=None, 
                attention_mask=None, 
                output_attentions=None, 
                output_hidden_states=None,
                return_dict=None, 
                labels=None, 
                anchor_features=None,
                positive_features=None,
                negative_features=None, 
                is_hybrid_loss=False,
                triplet_loss_p2=False,
                triplet_loss_cosine=False,
                use_mnr_loss_for_hybrid=False,  # Flag to choose MNRLoss for hybrid loss
                alpha=0.3, 
                margin=1.0):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss = None
        logits = None
        triplet_loss = None
        # Scenario 1: Classification
        if input_features is not None:
            outputs = self.wav2vec2bert(
                input_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs.last_hidden_state
            hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

            logits = self.classifier(hidden_states)

            if labels is not None:
                # Initialize the appropriate loss function only when required
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

        # Scenario 2: Hybrid Loss (Classification + Triplet or MNR Loss)
        elif is_hybrid_loss:
            # Forward pass for anchor, positive, and negative samples
            anchor_outputs = self.wav2vec2bert(anchor_features, return_dict=return_dict)
            anchor_hidden_states = self.merged_strategy(anchor_outputs.last_hidden_state, mode=self.pooling_mode)
            anchor_logits = self.classifier(anchor_hidden_states)

            positive_outputs = self.wav2vec2bert(positive_features, return_dict=return_dict)
            positive_hidden_states = self.merged_strategy(positive_outputs.last_hidden_state, mode=self.pooling_mode)
            positive_logits = self.classifier(positive_hidden_states)

            negative_outputs = self.wav2vec2bert(negative_features, return_dict=return_dict)
            negative_hidden_states = self.merged_strategy(negative_outputs.last_hidden_state, mode=self.pooling_mode)
            negative_logits = self.classifier(negative_hidden_states)

            # Classification loss (binary cross-entropy)
            ce_loss_fn = CrossEntropyLoss()  # Initialize CrossEntropy loss for classification
        
            # Move labels to the same device as logits
            labels = labels.to(anchor_logits.device)
            pos_labels = labels.to(positive_logits.device)  # Ensure pos_labels match positive_logits device

            # neg_labels = 1 - labels  # Assuming binary classification with labels 0 and 1
            neg_labels = torch.zeros(negative_logits.size(0), dtype=torch.long, device=negative_logits.device)

            ce_anchor = ce_loss_fn(anchor_logits, labels)
            ce_pos = ce_loss_fn(positive_logits, pos_labels)
            ce_neg = ce_loss_fn(negative_logits, neg_labels )
            ce_total = (ce_anchor + ce_pos + ce_neg) / 3.0

            # Embedding loss (either TripletMarginLoss or MNRLoss)
            if use_mnr_loss_for_hybrid:
                # Initialize MNRLoss only when required
                mnr_loss_fn = MNRLoss()
                embedding_loss = mnr_loss_fn(anchor_hidden_states, positive_hidden_states, negative_hidden_states)
                
            else:
                # Initialize TripletMarginLoss only when required
                if triplet_loss_p2:

                    triplet_loss_fn = TripletMarginLoss(margin=margin, p=2)

                elif triplet_loss_cosine:
                    
                    triplet_loss_fn = TripletMarginWithDistanceLoss(margin=margin,distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

                embedding_loss = triplet_loss_fn(anchor_hidden_states, positive_hidden_states, negative_hidden_states)
                triplet_loss = embedding_loss
            
            # Combine losses

            loss = alpha * ce_total + (1 - alpha) * embedding_loss

            # Stack vertically all these features for classification
            combined_hidden_states = torch.cat((anchor_hidden_states, positive_hidden_states, negative_hidden_states), dim=0)
            logits = self.classifier(combined_hidden_states)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            triplet_loss=triplet_loss,  # Include triplet loss for monitoring
            hidden_states=outputs.last_hidden_state if input_features is not None else combined_hidden_states,
            attentions=outputs.attentions if input_features is not None else None,
        )
