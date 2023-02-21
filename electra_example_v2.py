#reference: https://github.com/smallbenchnlp/ELECTRA-DeBERTa


######################################################################### Example -> ##########################################
# (1) instantiate the generator and discriminator
#     making sure that the generator is roughly a quarter to a half of the size of the discriminator
generator = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 256,              # smaller hidden dimension
    heads = 4,              # less heads
    ff_mult = 2,            # smaller feed forward intermediate dimension
    dim_head = 64,
    depth = 12,
    max_seq_len = 1024
)

discriminator = ReformerLM(
    num_tokens = 20000,
    emb_dim = 128,
    dim = 1024,
    dim_head = 64,
    heads = 16,
    depth = 12,
    ff_mult = 4,
    max_seq_len = 1024
)

# (2) weight tie the token and positional embeddings of generator and discriminator
generator.token_emb = discriminator.token_emb
generator.pos_emb = discriminator.pos_emb
# weight tie any other embeddings if available, token type embeddings, etc.

# (3) instantiate electra
trainer = Electra(
    generator,
    discriminator,
    discr_dim = 1024,           # the embedding dimension of the discriminator
    discr_layer = 'reformer',   # the layer name in the discriminator, whose output would be used for predicting token is still the same or replaced
    mask_token_id = 2,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    mask_ignore_token_ids = []  # ids of tokens to ignore for mask modeling ex. (cls, sep)
)

# (4) train

data = torch.randint(0, 20000, (1, 1024))

results = trainer(data)
results.loss.backward()

# after much training, the discriminator should have improved

torch.save(discriminator, f'./pretrained-model.pt')

#################################################### args -> ##########################################
# small_generator
{
    "attention_probs_dropout_prob": 0.1,
    "embedding_size": 128,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "relative_attention": true,
	"pos_att_type": "c2p|p2c",
    "layer_norm_eps": 1e-7,
    "max_position_embeddings": 128,
    "model_type": "deberta",
    "max_relative_positions": -1,
    "position_biased_input":false,
    "num_attention_heads": 4,
    "num_hidden_layers": 12,
    "type_vocab_size": 2,
    "vocab_size": 30522
}

# small_discriminator
{
    "attention_probs_dropout_prob": 0.1,
    "embedding_size": 128,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "relative_attention": true,
	"pos_att_type": "c2p|p2c",
    "layer_norm_eps": 1e-7,
    "max_position_embeddings": 128,
    "model_type": "deberta",
    "max_relative_positions": -1,
    "position_biased_input":false,
    "num_attention_heads": 4,
    "num_hidden_layers": 12,
    "type_vocab_size": 2,
    "vocab_size": 30522
}

#################################################### main structure in trainer -> ##########################################
generator = DebertaForMaskedLM(AutoConfig.from_pretrained(args.model_generator))
discriminator = DebertaForPreTraining(DebertaConfig.from_pretrained(args.model_discriminator))

tie_weights(generator, discriminator)
 
model = Electra(
    LogitsAdapter(generator),
    LogitsAdapter(discriminator),
    num_tokens = vocab_size,
    mask_token_id = mask_token_id,
    pad_token_id = pad_token_id,
    mask_prob = args.model_mask_prob,
    mask_ignore_token_ids = [tokenizer.vocab['[CLS]'], tokenizer.vocab['[SEP]']],
    random_token_prob = 0.0).to(device))  

##################################################### tie_weights #####################################################
def tie_weights(generator, discriminator):
    generator.deberta.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
    generator.deberta.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
    generator.deberta.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings
 

##################################################### DebertaForMaskedLM #####################################################
class DebertaForMaskedLM(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        self.cls = DebertaOnlyMLMHead(config)  # see below

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # normal deberta
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
      
      
# copied from transformers.models.bert.BertOnlyMLMHead with bert -> deberta
class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaLMPredictionHead(config) # see below

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
       
# copied from transformers.models.bert.BertLMPredictionHead with bert -> deberta
class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaPredictionHeadTransform(config) # see below

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
      
# copied from transformers.models.bert.BertPredictionHeadTransform with bert -> deberta
class DebertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

      
      
      
##################################################### DebertaForPreTraining #####################################################
class DebertaForPreTraining(DebertaPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)

        self.electra = DebertaModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config) # see below
        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see :obj:`input_ids`
            docstring) Indices should be in ``[0, 1]``:
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        Returns:
        Examples::
            >>> from transformers import ElectraTokenizer, ElectraForPreTraining
            >>> import torch
            >>> tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
            >>> model = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> logits = model(input_ids).logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # normal deberta
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output
        
        # see below
        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
  
  

class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits
      
@dataclass
class ElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.ElectraForPreTraining`.
    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None    

##########################################################################################################################
    
