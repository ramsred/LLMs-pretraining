import torch
import torch.nn as nn
from .classification_head import ClassificationHead, SpeechClassifierOutput
from transformers import Wav2Vec2BertModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class Wav2Vec2BERT(nn.Module):
    def __init__(self, model_name, pooling_mode='mean'):
        super().__init__()
        self.num_labels = 2
        self.pooling_mode = pooling_mode
        self.wav2vec2bert = Wav2Vec2BertModel.from_pretrained(model_name)
        self.config = self.wav2vec2bert.config
        self.classifier = ClassificationHead(self.wav2vec2bert.config)

    def merged_strategy(self,hidden_states,mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(self,input_features,attention_mask=None,output_attentions=None,output_hidden_states=None,return_dict=None,labels=None,):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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

        loss = None
        if labels is not None:
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

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

# import torch
# import torch.nn as nn
# from transformers import Wav2Vec2BertModel
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


# class Wav2Vec2BERT(nn.Module):
#     def __init__(self, model_name, pooling_mode='mean'):
#         super().__init__()
#         self.num_labels = 2
#         self.pooling_mode = pooling_mode
#         self.wav2vec2bert = Wav2Vec2BertModel.from_pretrained(model_name)
#         self.config = self.wav2vec2bert.config
#         self.classifier = ClassificationHead(self.wav2vec2bert.config)

#     def merged_strategy(self, hidden_states, mode="mean"):
#         if mode == "mean":
#             outputs = torch.mean(hidden_states, dim=1)
#         elif mode == "sum":
#             outputs = torch.sum(hidden_states, dim=1)
#         elif mode == "max":
#             outputs = torch.max(hidden_states, dim=1)[0]
#         else:
#             raise ValueError(
#                 "Pooling mode must be one of ['mean', 'sum', 'max']"
#             )
#         return outputs

#     def forward(self, input_features, attention_mask=None):
#         # Forward pass through Wav2Vec2BERT
#         outputs = self.wav2vec2bert(
#             input_features,
#             attention_mask=attention_mask,
#             return_dict=True,
#         )
#         hidden_states = outputs.last_hidden_state

#         # Apply pooling strategy
#         hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

#         # Classification layer
#         logits = self.classifier(hidden_states)

#         # Return logits tensor (compatible with Opacus)
#         return logits


# import torch
# import torch.nn as nn
# from transformers import Wav2Vec2BertModel


# class ClassificationHead(nn.Module):
#     """Head for wav2vec classification task."""

#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(0.1)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(self, features, **kwargs):
#         x = features
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


# class Wav2Vec2BERT(nn.Module):
#     def __init__(self, model_name, pooling_mode='mean'):
#         super().__init__()
#         self.num_labels = 2
#         self.pooling_mode = pooling_mode
#         self.wav2vec2bert = Wav2Vec2BertModel.from_pretrained(model_name)
#         self.config = self.wav2vec2bert.config
#         self.classifier = ClassificationHead(self.wav2vec2bert.config)

#     def merged_strategy(self, hidden_states, mode="mean"):
#         if mode == "mean":
#             outputs = torch.mean(hidden_states, dim=1)
#         elif mode == "sum":
#             outputs = torch.sum(hidden_states, dim=1)
#         elif mode == "max":
#             outputs = torch.max(hidden_states, dim=1)[0]
#         else:
#             raise ValueError(
#                 "Pooling mode must be one of ['mean', 'sum', 'max']"
#             )
#         return outputs

#     def forward(self, input_features, attention_mask=None):
#         # Forward pass through Wav2Vec2BERT
#         outputs = self.wav2vec2bert(
#             input_features=input_features,
#             attention_mask=attention_mask,
#             return_dict=True,
#         )
#         hidden_states = outputs.last_hidden_state

#         # Apply pooling strategy
#         hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

#         # Classification layer
#         logits = self.classifier(hidden_states)

#         # Return logits tensor (compatible with Opacus)
#         return logits
