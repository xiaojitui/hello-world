#! pip install datasets transformers

# Make sure your version of Transformers is at least 4.11.0 since the functionality was introduced in that version
import transformers
print(transformers.__version__)


# Preparing the dataset
from datasets import load_dataset
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
# You can replace the dataset above with any dataset hosted on the hub or use your own files. 
# Just uncomment the following cell and replace the paths with values that will lead to your files

datasets = load_dataset("text", data_files={"train": path_to_train.txt, "validation": path_to_validation.txt})

datasets["train"][10] # {'text': xxx}

# MLM
# we will randomly mask some tokens (by replacing them by [MASK]) and the labels will be adjusted to only include the masked tokens
# (we don't have to predict the non-masked tokens).

model_checkpoint = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["text"])

# block_size = tokenizer.model_max_length
block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
  

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
tokenized_datasets["train"][1] 
# {'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1],
# 'input_ids': [796, 569, 18354, 7496, 17740, 6711, 796, 220, 198]}

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# tokenizer.decode(lm_datasets["train"][1]["input_ids"])

from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-finetuned-wikitext2",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)


from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()


eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# Share
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("sgugger/my-awesome-model")
