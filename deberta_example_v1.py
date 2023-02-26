# refer: https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/trainer.py

# pesduo-code
d_emb = g_emb.detach() + delta_emb
total_loss = g_loss + 50 * d_loss
1. forward_r1 -> on g only -> g_out (same as d_in) 
2. backward_r1 (only use g_loss) -> update g (g_emb also update)
3. forward_r2 -> on d only -> d_out 
4. backward_r2 (only use d_loss) -> update d (only delta_emb update)

def model():
	def __init__(self, lambda_p = 50.0):
		self.generator_model = generator
		self.discriminator = discriminator
		self.delta_emb = torch.zero_like(self.g_emb)
		self.lambda_p = lambda_p
		#self.tie_generator_and_discriminator_embeddings()
		
	def tie_generator_and_discriminator_embeddings(self):
        self.discriminator_model.set_input_embeddings(
            self.generator_model.get_input_embeddings().detach() + self.delta_emb
        )
	
	def forward(self, inputs, labels, 
				attention_mask=None, 
				token_type_ids=None, 
				update_part = 'generator'):
		#
		if update_part == 'generator':
			d_inputs = inputs.clone()
			# run masked LM.
			g_out = self.generator_model(
				inputs,
				labels=labels,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids,
				output_hidden_states=True,
			)

			# get samples from masked LM.
			sample_probs = torch.softmax(g_out[1], dim=-1, dtype=torch.float32)
			sample_probs = sample_probs.view(-1, self.vocab_size)

			sampled_tokens = torch.multinomial(sample_probs, 1).view(-1)
			sampled_tokens = sampled_tokens.view(d_inputs.shape[0], -1)

			# labels have a -100 value to mask out loss from unchanged tokens.
			mask = labels.ne(-100)

			# replace the masked out tokens of the input with the generator predictions.
			d_inputs[mask] = sampled_tokens[mask]
			# print(len(g_out.hidden_states))
			# print(len(g_out))
			# print(d_inputs[mask])
			# raj
			# turn mask into new target labels.  1 (True) for corrupted, 0 otherwise.
			# if the prediction was correct, mark it as uncorrupted.
			correct_preds = sampled_tokens == labels
			d_labels = mask.long()
			d_labels[correct_preds] = 0
			# g_out[0] is loss, g_out[1] is logits
			#g_loss = g_out.loss
			return g_out, d_inputs, d_labels
		#
		elif update_part = 'discriminator'
			# run token classification, predict whether each token was corrupted.
			self.tie_generator_and_discriminator_embeddings()
			d_out = self.discriminator_model(
				inputs,
				labels,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids,
			)
			#d_loss = d_out.loss * self.lambda_p 
			return d_out
		else:
			raise ValueError('**not supported**')


def train():
	model = model(args)
	optimizer = torch.optim.AdamW(model.parameters())
	# 1a. clear
	model.zero_grad()
	# 1b. update generator
	g_out, d_inputs, d_labels = model(inputs, labels, update_part = 'generator')
	g_loss = g_out.loss
	g_loss.backward()
	optimizer.step()
	
	# 2a. clear
	model.zero_grad()
	# 2b. update discriminator
	d_out = model(d_inputs, d_labels, update_part = 'discriminator')
	d_loss = d_out.loss * model.lambda_p
	d_loss.backward()
	optimizer.step()
	
	# 3. save all
	total_loss = g_loss + d_loss
	return total_loss


def eval()
	model = model(args)
	with torch.no_grad():
		g_out, d_inputs, d_labels = model(inputs, labels, update_part = 'generator')
		d_out = model(d_inputs, d_labels, update_part = 'discriminator')
	return d_out
	
	



############################ transformer -> trainer #############################
#
L1464 -> Trainer

Trainer:
	inner_training_loop = find_executable_batch_size # see below
	(
		self._inner_training_loop, # see below
		self._train_batch_size, # from arg
		args.auto_find_batch_size # from arg
        )
    return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )
		

# 
find_executable_batch_size
here, just do:
return functools.partial(function, batch_size=starting_batch_size)
A basic decorator that will try to execute `function`.


# L1550
_inner_training_loop
	...
	# L1789
	tr_loss_step = self.training_step(model, inputs)
	
	
	

# L2513
training_step
	# 2535
	smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
	# 2548
	loss.backward()
	# 2561
	compute_loss(self, model, inputs, return_outputs=False)
	




#######################################################
