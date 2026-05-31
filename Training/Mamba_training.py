# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# pip install causal-conv1d==1.4.0 && pip install mamba-ssm==2.2.2
# pip install "huggingface_hub<1.0" "datasets>=3.0.0" --force-reinstall
# pip install evaluate
# pip install rouge_score
# pip install bert_score

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("Fast Path (CUDA) is AVAILABLE.")
except ImportError:
    print("Fast Path is MISSING. Running in slow Python mode.")


import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


# 1. Configuration

MODEL_ID = "state-spaces/mamba-130m-hf"
DATASET_ID = "s-nlp/paradetox"
OUTPUT_DIR = "./mamba_detox"
MAX_LENGTH = 128


# 2. Prepare Data with "STOP TOKEN"
def format_data(example):

    return {
        "text": f"### Toxic: {example['en_toxic_comment']}\n### Neutral: {example['en_neutral_comment']} ### END"
    }

def load_and_process_data(tokenizer):
    print(f"Loading FULL dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train")
    print(f"Using full dataset: {len(dataset)} examples")

    print("Formatting prompts with ### END marker...")
    dataset = dataset.map(format_data)

    def tokenize_function(examples):
        texts = [t + tokenizer.eos_token for t in examples["text"]]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["en_toxic_comment", "en_neutral_comment", "text"])

    split_dataset = tokenized_datasets.train_test_split(test_size=0.1)
    return split_dataset["train"], split_dataset["test"]


# 3. Main Training Loop

def main():
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected.")
    else:
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Model: {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float32
    )

    train_dataset, eval_dataset = load_and_process_data(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=False,
        push_to_hub=False,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting Training...")
    trainer.train()

    print(f"Training finished. Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


# 4. Inference Function

def inference(prompt_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, torch_dtype=torch.float32).to(device)

    input_text = f"### Toxic: {prompt_text}\n### Neutral:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


    if "### Neutral:" in generated_text:
        result = generated_text.split("### Neutral:")[1]
        result = result.split("### END")[0].strip()
    else:
        result = generated_text

    return result

if __name__ == "__main__":
    main()

    print("\n--- Quality Test ---")
    samples = ["You are so stupid.", "I hate you.", "Shut up idiot."]
    for s in samples:
        print(f"Toxic: {s}")
        print(f"Fixed: {inference(s)}\n")