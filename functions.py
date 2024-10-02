from nltk.tokenize import sent_tokenize
import spacy
import custom_spacy as csp
import pysbd
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
import json
import re

def open_file(file_path, type):
    
    with open(file_path, 'r', encoding="utf-8") as f:
        if (type == "json"):
            return json.load(f)
        elif (type == "txt"):
            return f.read()

def sent_segmentation(document, method='nltk'):
    """Segmentation of the document as sentences using the specified method.
    
    Args:
        document (str): The document to segment.
        method (str): The method to use for segmentation ('nltk', 'spacy', 'custom_spacy' or 'pySBD').
        
    Returns:
        List[str]: A list of tokenized sentences.
    """
    if method == 'nltk':
        return sent_tokenize(document)
    elif method == 'spacy':
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp.add_pipe("sentencizer") 
        split_doc = split(document)
        sentences = []
        for chunk in split_doc:
            chunk = re.sub(r'\"', '', chunk)    # remove double quote because error
            doc = nlp(chunk)
            for sent in doc.sents:
                sentences.append(sent.text)
        return sentences
    elif method == 'custom_spacy':
        nlp = csp.custom_spacy_model()
        split_doc = split(document)
        sentences = []
        for chunk in split_doc:
            chunk = re.sub(r'\"', '', chunk)    # remove double quote because error
            doc = nlp(chunk)
            for sent in doc.sents:
                sentences.append(sent.text)
        return sentences
    elif method == 'pySBD':        
        seg = pysbd.Segmenter(language="en", clean=False)
        return seg.segment(document)
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
    
def bb25LegalSum(sentences, model_name="bert-base-uncased", n_clusters = 5):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    sentence_embeddings = get_sentence_embeddings(sentences, tokenizer, model)
      
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(sentence_embeddings)

    # Étiquettes des clusters
    labels = kmeans.labels_
    
    cluster = {}
    for i in range(n_clusters):
        #print(f"\nCluster {i+1}:")
        cluster[i] = []
        for j, sentence in enumerate(sentences):
            if labels[j] == i:
                #print(f"- {sentence}")
                cluster[i].append(sentence)
                
    silhouette_avg = silhouette_score(sentence_embeddings, labels)
    #print(f"\nSilhouette Score: {silhouette_avg}")
    
    tokenized_clusters = {}

    # Tokenisation des documents pour chaque cluster
    for i, sentences in cluster.items():
        tokenized_clusters[i] = [word_tokenize(sentence.lower()) for sentence in sentences]

    # Initialiser un modèle BM25 pour chaque cluster
    bm25_models = {}
    for i, tokenized_docs in tokenized_clusters.items():
        bm25_models[i] = BM25Okapi(tokenized_docs)
    
    query = "law and legal rights"
    tokenized_query = word_tokenize(query.lower())   
     
    for cluster_id, bm25 in bm25_models.items():
        # Calcul des scores pour la requête dans chaque cluster
        scores = bm25.get_scores(tokenized_query)
        
        
    best_sentences = []

    for cluster_id, bm25 in bm25_models.items():
            # Récupérer les phrases les plus pertinentes pour la requête dans ce cluster
            top_docs = bm25.get_top_n(tokenized_query, tokenized_clusters[cluster_id], n=2)

            # Extraire les phrases pertinentes et les ajouter à la liste
            for doc in top_docs:
                sentence = ' '.join(doc)  # Convertir le tokenized doc en phrase
                best_sentences.append(sentence)

    return best_sentences
    

    
def get_sentence_embeddings(sentences, tokenizer, model):
    """Obtenir les embeddings de phrases avec BERT
    Args:
        sentences (List[str]): Liste des phrases à encoder
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer BERT
        model (transformers.PreTrainedModel): Modèle BERT
    Returns:
        np.array: Tableau des embeddings de phrases    
    """
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())  # Moyenne des embeddings
    return np.array(embeddings)

def split(text, max_length=3530): 
    split_text = text.split('\n')
    result = []
    
    for chunk in split_text:
        while len(chunk) > max_length:
            sub_chunk = chunk[:max_length]
            last_period_position = sub_chunk.rfind('.')
            
            if last_period_position == -1:
                last_period_position = max_length
            
            if chunk[:last_period_position+1].strip():
                result.append(chunk[:last_period_position+1].strip())
            chunk = chunk[last_period_position+1:].strip()
        
        if chunk and chunk.strip():
            result.append(chunk.strip())
                
    return result

    
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