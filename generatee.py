from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load tokenizer and model once
model_path = "./saved_model/final model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

def generate_text(
    prompt,
    max_new_tokens=100,  # Control how much new text to generate
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.5,
    do_sample=True,
    num_beams=1
):
    # Tokenize the input prompt, allowing truncation and padding to the left
    inputs = tokenizer(prompt.strip(), return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate text from the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Ensure attention_mask is passed
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,  # Generate a specific number of new tokens
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=2,
            do_sample=do_sample,
            num_beams=num_beams
        )

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


