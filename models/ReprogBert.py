from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertPreTrainedModel, BertEmbeddings, BertModel, BertOnlyMLMHead
from torch import nn
import torch
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss


class BertConfigProtein(BertConfig):
    def __init__(self, vocab_size_protein=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size_protein = vocab_size_protein


class ModifiedBertLMPredictionHead(BertLMPredictionHead):
    def __init__(self, config):
        super().__init__(config)
        self.gamma = nn.Linear(config.vocab_size, config.vocab_size_protein, bias=False)
        self.gammabias = nn.Parameter(torch.zeros(config.vocab_size_protein))
        self.gamma.bias = self.gammabias

    def forward(self, hidden_states):
        hidden_states = super().forward(hidden_states)
        hidden_states = self.gamma(hidden_states)
        return hidden_states


class ModifiedBertOnlyMLMHead(BertOnlyMLMHead):
    def __init__(self, config):
        super().__init__(config)
        self.predictions = ModifiedBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class ModifiedBertPreTrainedModel(BertPreTrainedModel):
    config_class = BertConfigProtein


class ModifiedBertEmbeddings(BertEmbeddings):

    def __init__(self, config):
        super().__init__(config)
        self.conf = config

        # CHANGED: Added theta projection matrix of input protein domain to english domain
        self.theta = nn.Embedding(config.vocab_size_protein, config.vocab_size, padding_idx=config.pad_token_id_prot)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # CHANGED: Project input ids, and then do the embeddings
        if inputs_embeds is None:
            inputs_embeds = self.theta(input_ids)
            inputs_embeds = torch.matmul(inputs_embeds, self.word_embeddings.weight)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ModifiedBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.embeddings = ModifiedBertEmbeddings(config)


class BertForMaskedLMProt(ModifiedBertPreTrainedModel, BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        # CHANGED: BertModel -> ModifiedBertModel
        self.bert = ModifiedBertModel(config, add_pooling_layer=False)

        # CHANGED: BertOnlyMLMHead -> ModifiedBertOnlyMLMHead
        self.cls = ModifiedBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # CHANGED: self.config.vocab_size -> self.config.vocab_size_protein
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size_protein), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )