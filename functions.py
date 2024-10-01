from nltk.tokenize import sent_tokenize
import spacy
import custom_spacy as csp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import pandas as pd

def sent_segmentation(document, method='nltk'):
    """Segmentation of the document as sentences using the specified method.
    
    Args:
        document (str): The document to segment.
        method (str): The method to use for segmentation ('nltk', 'spacy', 'custom_spacy').
        
    Returns:
        List[str]: A list of tokenized sentences.
    """
    if method == 'nltk':
        return sent_tokenize(document)
    elif method == 'spacy':
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp.add_pipe("sentencizer") 
        doc = nlp(document)
        return [sent.text for sent in doc.sents]
    elif method == 'custom_spacy':
        nlp = csp.custom_spacy_model()
        doc = nlp(document)
        return [sent.text for sent in doc.sents]
    else:
        raise ValueError("Unsupported tokenization method. Choose 'nltk', 'spacy', or 'custom_spacy'.")

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
    
def evaluation(text, ref):
    rouges = rouge_evaluations(text, ref)
    
def rouge_evaluations(text, ref):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(text, ref)

    return rouge_to_df(scores)

def rouge_to_df(scores):
    
    data = {
        'Metric': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }

    for metric, score in scores.items():
        data['Metric'].append(metric)
        data['Precision'].append(score.precision)
        data['Recall'].append(score.recall)
        data['F1-Score'].append(score.fmeasure)

    return pd.DataFrame(data)