# reference: https://github.com/retarfi/language-pretraining

########################################################### config
"electra-base" : {
    "number-of-layers" : 12,
    "hidden-size" : 768,
    "sequence-length" : 512,
    "ffn-inner-hidden-size" : 3072,
    "attention-heads" : 12,
    "embedding-size" : 768,
    "generator-size" : "1/3",
    "mask-percent" : 15,
    "warmup-steps" : 10000,
    "learning-rate" : 2e-4,
    "batch-size" : {
        "-1" : 256
    },
    "train-steps" : 766000
},


########################################################### main function -> 
 run_pretraining(
        tokenizer = tokenizer,
        dataset_dir = args.dataset_dir,
        model_name = model_name,
        model_dir = args.model_dir,
        load_pretrained = load_pretrained,
        param_config = param_config[args.model_type],
        fp16_type = args.fp16_type,
        do_whole_word_mask = args.do_whole_word_mask,
        do_continue = args.do_continue,
        node_rank = args.node_rank,
        local_rank = args.local_rank,
        run_name = args.run_name,
    )

    
########################################################### get electra ->  
def get_model_electra(
        tokenizer:PreTrainedTokenizerBase,
        load_pretrained:bool,
        param_config:dict,
    ) -> PreTrainedModel:

    if load_pretrained:
        model = utils.ElectraForPretrainingModel.from_pretrained(
            param_config['pretrained_generator_model_name_or_path'],
            param_config['pretrained_discriminator_model_name_or_path']
        )
    else:
        frac_generator = Fraction(param_config['generator-size']) # 1/3, 1/1, 1/4
        config_generator = ElectraConfig(
            pad_token_id = tokenizer.pad_token_id,
            vocab_size = tokenizer.vocab_size, 
            embedding_size = param_config['embedding-size'],
            hidden_size = int(param_config['hidden-size'] * frac_generator), 
            num_attention_heads = int(param_config['attention-heads'] * frac_generator),
            num_hidden_layers = param_config['number-of-layers'],
            intermediate_size = int(param_config['ffn-inner-hidden-size'] * frac_generator),
            max_position_embeddings = param_config['sequence-length'],
        )
        config_discriminator = ElectraConfig(
            pad_token_id = tokenizer.pad_token_id,
            vocab_size = tokenizer.vocab_size, 
            embedding_size = param_config['embedding-size'],
            hidden_size = param_config['hidden-size'], 
            num_attention_heads = param_config['attention-heads'],
            num_hidden_layers = param_config['number-of-layers'],
            intermediate_size = param_config['ffn-inner-hidden-size'],
            max_position_embeddings = param_config['sequence-length'],
        )
        model = utils.ElectraForPretrainingModel(
            config_generator = config_generator,
            config_discriminator = config_discriminator,
        )
    return model


########################################################### get electra ->  
from transformers.models.electra.modeling_electra import ElectraForPreTrainingOutput
from transformers.file_utils import ModelOutput
@dataclass
class MyElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.ElectraForPreTraining`.
    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA objective.
        gen_loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Generator loss of the ELECTRA objective.
        disc_loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Discriminator loss of the ELECTRA objective.
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
    gen_loss: Optional[torch.FloatTensor] = None
    disc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
        
        
# 
class ElectraForPretrainingModel(ElectraPreTrainedModel):
    def __init__(self, config_generator, config_discriminator, loss_weights=(1.0,50.0)):
        super().__init__(config_discriminator)

        self.generator = ElectraForMaskedLM(config_generator)
        self.discriminator = ElectraForPreTraining(config_discriminator)
        # weight sharing
        self.discriminator.electra.embeddings = self.generator.electra.embeddings
        self.generator.generator_lm_head.weight = self.generator.electra.embeddings.word_embeddings.weight
        self.init_weights()
        self.loss_weights = loss_weights
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.) 
        # Using gumbel softmax to sample generations from geneartor as input of discriminator
    
    def to(self, *args, **kwargs):
        "Also set dtype and device of contained gumbel distribution if needed"
        return_object = super().to(*args, **kwargs)
        device, dtype = self.generator.device, torch.float32
        # https://github.com/pytorch/pytorch/issues/41663
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))
        return return_object

    def forward(
            self, 
            input_ids, labels, attention_mask=None, token_type_ids=None, 
            position_ids=None,
            output_attentions=None, 
            output_hidden_states=None, return_dict=None
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_gen = self.generator(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids,
            labels=labels, output_attentions=False, 
            output_hidden_states=False, return_dict=True
        )

        loss_gen = outputs_gen.loss # (1,)
        if torch.isnan(loss_gen):
            raise ValueError('output_gen is NaN')
        logits_gen = outputs_gen.logits # (batch_size, seq_length, config.vocab_size)
        with torch.no_grad():
            masked_bool = (labels != -100) # which are masked
            # ids_answer = input_ids.clone()
            # ids_answer[masked_bool] = labels[masked_bool]

            logits = logits_gen[masked_bool] # pick masked ones
            gumbel = self.gumbel_dist.sample(logits.shape) # randomness in picking generators
            tokens_replaced = (logits + gumbel).argmax(dim=-1) # this is my best prediction
            input_ids_disc = input_ids.clone() # raw ids
            input_ids_disc[masked_bool] = tokens_replaced # replace the 'masks' with the 'predictions'
            labels_disc = torch.zeros(labels.shape, dtype=torch.long, device=labels.device) # label for discriminator
            labels_disc[masked_bool] = (tokens_replaced != labels[masked_bool]).to(torch.long) # 0, 1; set replaced to 1, original to 0

        outputs_disc = self.discriminator(
            input_ids=input_ids_disc, attention_mask=attention_mask, token_type_ids=token_type_ids, 
            position_ids=position_ids,
            labels=labels_disc, output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        if not return_dict:
            loss_disc = outputs_disc[0]
            total_loss = self.loss_weights[0] * loss_gen + self.loss_weights[1] * loss_disc
            return ((total_loss, loss_gen.detach(), loss_disc.detach()) + outputs_disc[1:]) if total_loss is not None else outputs_disc

        loss_disc = outputs_disc.loss
        if torch.isnan(loss_disc):
            raise ValueError('loss_disc is NaN')
        total_loss = self.loss_weights[0] * loss_gen + self.loss_weights[1] * loss_disc
        return MyElectraForPreTrainingOutput(
            loss=total_loss,
            gen_loss=loss_gen.detach(),
            disc_loss=loss_disc.detach(),
            logits=outputs_disc.logits,
            hidden_states=outputs_disc.hidden_states,
            attentions=outputs_disc.attentions,
        )
    
########################################################### data collator ->
# Override transformers.data.data_collator.DataCollatorForWholeWordMask
@dataclass
class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(
        self,
        tokenizer:PreTrainedTokenizer,
        mlm:bool = True,
        mlm_probability:float = 0.15,
        rate_replaced:float = 0.8,
        rate_random:float = 0.1,
        rate_unchanged:float = 0.1
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.rate_replaced = rate_replaced
        rate_random= rate_random
        rate_unchanged = rate_unchanged
        assert self.rate_replaced + rate_random + rate_unchanged == 1

        # 逆算でreplaceされてないもののうちrandomにする割合を求める
        self.rate_random_condition = self.mlm_probability * rate_random / (1 - self.mlm_probability * self.rate_replaced)
        # 逆算でreplaceとrandomでないもののうち何の処理もされないものの割合を求める
        self.rate_left_condition = (1 - self.mlm_probability) / (1 - self.mlm_probability * (self.rate_replaced + rate_random))

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        # cast to torch.long
        for k, v in batch.items():
            batch[k] = v.to(torch.long)
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # inputs:(batch, seq_len)
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = self._whole_word_mask(labels, special_tokens_mask)
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, self.rate_random_condition)).bool() & ~indices_replaced & ~special_tokens_mask
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        indices_left = torch.bernoulli(torch.full(labels.shape, self.rate_left_condition)).bool() & ~indices_replaced & ~indices_random & ~special_tokens_mask

        labels[(~indices_replaced) & (~indices_random) & (~indices_left)] = -100  # We only compute loss on masked tokens
        return inputs, labels


    def _whole_word_mask(
        self, input_ids:torch.Tensor, special_tokens_mask:torch.Tensor,
        max_predictions:int=512
    ) -> torch.Tensor:
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        # 0.15をboolにするときにsubwordを考慮すべき
        # idea: 候補となる単語をmain-wordとして抽出→shuffle→前から0.15*0.8になるまで抽出
        # →これらをmask
        mask_indices = []
        for document_ids, special_tokens in zip(input_ids.tolist(), special_tokens_mask.tolist()):
            document_tokens = self.tokenizer.convert_ids_to_tokens(document_ids)
            cand_indexes = []
            for i, (token, is_special_token) in enumerate(zip(document_tokens, special_tokens)):
                if is_special_token:
                    continue
                if token == self.tokenizer.pad_token:
                    break
                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])

            np.random.shuffle(cand_indexes)
            num_to_mask = min(max_predictions, max(1, int(round(len(document_tokens) * self.mlm_probability * self.rate_replaced))))
            total_masked = 0
            covered_indexes = set()
            for index_set in cand_indexes:
                if total_masked >= num_to_mask:
                    break
                # If adding a whole-word mask would exceed the maximum number of
                # predictions, then just skip this candidate.
                if total_masked + len(index_set) > num_to_mask:
                    continue
                covered_indexes = covered_indexes | set(index_set)
                total_masked += len(index_set)

            mask = [1 if i in covered_indexes else 0 for i in range(len(document_ids))]
            mask_indices.append(mask)
        mask_indices = torch.tensor(mask_indices).bool()
        return mask_indices


@dataclass
class DataCollatorForLanguageModelingWithElectra(DataCollatorForLanguageModeling):
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.85)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # The rest of the time (15% of the time) we keep the masked input tokens unchanged
        return inputs, labels


########################################################### main function ->  
def run_pretraining(
        tokenizer:PreTrainedTokenizerBase, 
        dataset_dir:str,
        model_name:str,
        model_dir:str,
        load_pretrained:bool,
        param_config:dict,
        fp16_type:int,
        do_whole_word_mask:bool,
        do_continue:bool,
        node_rank:int,
        local_rank:int,
        run_name:str
    ) -> None:
    
    # initialize
    training_args = TrainingArguments(
        output_dir = model_dir,
        do_train = True,
        do_eval = False, # default
        per_device_train_batch_size = per_device_train_batch_size,
        learning_rate = param_config['learning-rate'], 
        adam_beta1 = 0.9, # same as BERT paper
        adam_beta2 = 0.999, # same as BERT paper
        adam_epsilon = 1e-6,
        weight_decay = 0.01, # same as BERT paper
        warmup_steps = param_config['warmup-steps'], 
        logging_dir = os.path.join(os.path.dirname(__file__), f"runs/{run_name}"),
        save_steps = param_config['save-steps'] if 'save-steps' in param_config.keys() else 50000, #default:500
        save_strategy = "steps", # default:"steps"
        logging_steps = param_config['logging-steps'] if 'logging-steps' in param_config.keys() else 5000, # default:500
        save_total_limit = 20, # optional
        seed = 42, # default
        fp16 = bool(fp16_type!=0),
        fp16_opt_level = f"O{fp16_type}", 
        #:"O1":Mixed Precision (recommended for typical use), "O2":“Almost FP16” Mixed Precision, "O3":FP16 training
        disable_tqdm = True,
        max_steps = param_config['train-steps'],
        gradient_accumulation_steps = 1 if 'accumulation-steps' not in param_config.keys() else param_config['accumulation-steps'],
        dataloader_num_workers = 3,
        dataloader_pin_memory=False,
        local_rank = local_rank,
        report_to = "tensorboard"
    )
    if not do_continue:
        if local_rank != -1:
            if torch.cuda.device_count() > 0:
                training_args.per_device_train_batch_size = int(param_config['batch-size'][str(node_rank)] / torch.cuda.device_count())
            else:
                training_args.per_device_train_batch_size = param_config['batch-size'][str(node_rank)]
        torch.save(training_args, os.path.join(model_dir, "training_args.bin"))
        
        
    # dataset
    dataset = datasets.load_from_disk(dataset_dir)
    dataset.set_format(type='torch')
    logger.info('Dataset is loaded')

    
    # for electra
    model = get_model_electra(tokenizer, load_pretrained, param_config)
    logger.info(f'{model_name} model is loaded')    
    
    # data collator
    mlm_probability = param_config['mask-percent']/100 # just 0.15
    if do_whole_word_mask:
        data_collator = utils.DataCollatorForWholeWordMask(
                tokenizer = tokenizer, 
                mlm = True,
                mlm_probability = mlm_probability,
                rate_replaced = 0.85,
                rate_random = 0,
                rate_unchanged = 0.15
            )
    else:
        data_collator = utils.DataCollatorForLanguageModelingWithElectra(
                tokenizer = tokenizer,
                mlm = True,
                mlm_probability = mlm_probability
            )
    logger.info('Datacollator was complete.')

    # train
    trainer = utils.MyTrainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = dataset,
        node_rank = node_rank
    )
    trainer.batch_config = param_config['batch-size']
    trainer.real_batch_size = sum(param_config['batch-size'].values())

    logger.info('Pretraining starts.')
    resume_from_checkpoint = True if do_continue else None
    trainoutput = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
        do_log_loss_gen_disc=bool(model_name == 'electra')
    )
    
  
########################################################### MyTrainer ->  
class MyTrainer(Trainer):
    def __init__(self, node_rank, **kwargs):
        super().__init__(**kwargs)
        self.node_rank = node_rank
# skip for now
# main change is to log generator & discriminator loss
###########################################################
