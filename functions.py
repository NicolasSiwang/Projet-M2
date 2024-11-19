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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import evaluate

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

def PCA_kmeans_optimal_clusters(sentence_embeddings, max_clusters=7):
    """
    Applique une analyse en composantes principales (PCA) et un K-Means 
    pour déterminer le nombre optimal de clusters.
    """
    # Réduire la dimensionnalité avec PCA (par exemple, à 2 ou 3 dimensions)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(sentence_embeddings)
    
    silhouette_scores = []
    k_values = list(range(2, max_clusters + 1))
    
    # Tester différents nombres de clusters
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(reduced_embeddings)
        score = silhouette_score(reduced_embeddings, labels)
        silhouette_scores.append(score)
    
    # Trouver la valeur de k avec le meilleur score de silhouette
    optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    
    # Afficher le graphe du score de silhouette
    # plt.figure()
    # plt.plot(k_values, silhouette_scores, marker='o')
    # plt.xlabel('Nombre de clusters (k)')
    # plt.ylabel('Score de silhouette')
    # plt.title('Détermination du nombre optimal de clusters')
    # plt.show()
    
    return optimal_k
    
def bb25LegalSum(sentences, model_name="bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    sentence_embeddings = get_sentence_embeddings(sentences, tokenizer, model)

    # Déterminer le nombre optimal de clusters
    print("Appel pca")
    n_clusters = PCA_kmeans_optimal_clusters(sentence_embeddings)
    print("fin pca")
      
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
    data = [
        {'Metric': metric, 'Precision': score.precision, 'Recall': score.recall, 'F1-Score': score.fmeasure}
        for metric, score in scores.items()
    ]
    return pd.DataFrame(data)

def bleu_evaluations(text, ref):
    metrics = {
        "bleu": evaluate.load("bleu"),
        "google_bleu": evaluate.load("google_bleu")
    }
    
    results = {
        name: metric.compute(predictions=text, references=ref).get(name, None)
        for name, metric in metrics.items()
    }
    return bleu_to_df(results)

def bleu_to_df(results):
    data = [{"Metric": metric, "Score": score} for metric, score in results.items()]
    return pd.DataFrame(data)


def highlight_html(full_text, extracts):
    '''The extrats is a list of fragments of full_text. Highlight them in full_text, which is an HTML.'''

    highlighted_text = full_text

    for extract in extracts:
        # Skip empty strings
        if extract.strip() == "":
            continue

        # Normalize the extract_text by collapsing whitespace
        normalized_extract = re.sub(r'\s+', ' ', extract.strip())

        # Find the first occurrence of the extract_text in the full_text
        pattern = re.compile(re.escape(normalized_extract), re.IGNORECASE)

        match = pattern.search(highlighted_text)
        if match:
            start_index, end_index = match.span()
            
            # Check if the match is within HTML tags (excluding 'a' and 'em' tags)
            if (highlighted_text.rfind('<', 0, start_index) == -1 or
                highlighted_text.rfind('>', 0, start_index) > highlighted_text.rfind('<', 0, start_index) or
                re.search(r'<(a|em)>', highlighted_text[start_index:end_index])):

                # Highlight the first occurrence of the text outside disallowed tags
                highlighted_text = (highlighted_text[:start_index] +
                                    f'<span style="background-color: yellow; color: black; font-weight: bold;">{match.group(0)}</span>' +
                                    highlighted_text[end_index:])

    return highlighted_text

def highlight_text(full_text, extracts):
    '''The extrats is a list of fragments of full_text. Highlight them in full_text, which is a text.'''

    # Highlight occurrences of each extract in the full text
    highlighted_text = full_text
    for extract in extracts:
        if extract:
            # This pattern handles the case where extracts may include new lines or spaces
            pattern = re.escape(extract).replace(r'\ ', r'\s*')  # Allow for variable whitespace
            
            # Use re.search to find the first occurrence
            match = re.search(pattern, highlighted_text, flags=re.IGNORECASE)
            if match:
                start_index, end_index = match.span()
                # Highlight the first occurrence
                highlighted_text = (highlighted_text[:start_index] +
                                    f'<span style="background-color: yellow; color: black;">{highlighted_text[start_index:end_index]}</span>' +
                                    highlighted_text[end_index:])

    return highlighted_text