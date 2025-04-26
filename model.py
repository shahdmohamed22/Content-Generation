from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_path="./saved_model/final model"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token 
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizers