from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def summarize(text, model_name="legal-pegasus", min_length=150, max_length=250):
    """Return a summary"""
    
    if (model_name == "legal-pegasus"):
        tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")  
        model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")
        input_tokenized = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
        
        summary_ids = model.generate(input_tokenized,
                                    num_beams=9,
                                    no_repeat_ngram_size=3,
                                    length_penalty=2.0,
                                    min_length=min_length,
                                    max_length=max_length,
                                    early_stopping=True)

        return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
    
    else:
        return "Model not available"
    