from nltk.tokenize import sent_tokenize
import spacy
import utilities.custom_spacy as csp
import pysbd

from nltk.tokenize import word_tokenize
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rank_bm25 import BM25Okapi
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from rouge_score import rouge_scorer
from bert_score import BERTScorer
import evaluate

import pandas as pd
import numpy as np
import json
import re

from transformers import logging

def open_file(file_path, type): 
    with open(file_path, 'r', encoding="utf-8", errors='replace') as f:
        if (type == "json"):
            return json.load(f)
        elif (type == "txt"):
            return f.read()

def sent_segmentation(document, method='pySBD'):
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
    
    # Tester différents nombres dew clusters
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

def bb25LegalSum(sentences, model_name="bert-base-uncased", query="law and legal rights", limit_output=True):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    unique_sentences = list(set(sentences))
    sentence_embeddings = get_sentence_embeddings(unique_sentences, tokenizer, model)

    # Déterminer le nombre optimal de clusters
    n_clusters = PCA_kmeans_optimal_clusters(sentence_embeddings)
      
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(sentence_embeddings)

    # Étiquettes des clusters
    unique_labels = kmeans.labels_
    sentence_to_label = {sentence: unique_labels[i] for i, sentence in enumerate(unique_sentences)}
    
    clusters = {i: [] for i in range(n_clusters)}
    for i, sentence in enumerate(sentences):
        label = sentence_to_label[sentence]
        clusters[label].append(sentence.replace('\xa0', ' '))
                
    silhouette_avg = silhouette_score(sentence_embeddings, unique_labels)
    # print(f"Silhouette Score: {silhouette_avg}")
    
    # Filtrer les clusters de phrases très courtes
    def is_valid_cluster(cluster_sentences, min_length=5):
        """Checks if a cluster is valid based on average sentence length."""
        avg_length = sum(len(sentence.split()) for sentence in cluster_sentences) / len(cluster_sentences)
        return avg_length >= min_length
    
    filtered_clusters = {
        cluster_id: cluster_sentences
        for cluster_id, cluster_sentences in clusters.items()
        if is_valid_cluster(cluster_sentences)
    }
    
    # Tokenization des clusters filtrés
    # tokenized_clusters = {
    #     cluster_id: [word_tokenize(sentence) for sentence in cluster_sentences]
    #     for cluster_id, cluster_sentences in filtered_clusters.items()
    # }
    
    tokenized_clusters = {}
    for cluster_id, cluster_sentences in filtered_clusters.items():
        tokenized_sentences = []
        for sentence in cluster_sentences:
            tokenized_sentences.append(word_tokenize(sentence))
        tokenized_clusters[cluster_id] = tokenized_sentences

    # Initialiser un modèle BM25 pour chaque cluster
    bm25_models = {
        cluster_id: BM25Okapi(tokenized_docs)
        for cluster_id, tokenized_docs in tokenized_clusters.items()
    }
    
    tokenized_query = word_tokenize(query)
    best_sentences = []
    total_tokens = 0

    for cluster_id, bm25 in bm25_models.items():
        # Récupérer les phrases les plus pertinentes pour la requête dans ce cluster
        top_docs = bm25.get_top_n(tokenized_query, clusters[cluster_id], n=10)  

        for doc in top_docs:
            token_count = len(word_tokenize(doc))
            # Si le total de token ne dépasse pas 512, on ajoute la phrase
            if total_tokens + token_count <= 512: 
                best_sentences.append(doc)
                # Limite la taille de sortie à sortie 512
                if limit_output:
                    total_tokens += token_count
            else:
                break

        # Si le total de token dépasse 512, on s'arrête
        if total_tokens >= 512:
            break
            
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
    """Split text into smaller chuncks and try to cut keep them as sentences"""
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

def contains_date_format(text):
    """Search for date"""
    pattern = r'\[\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b\]'
    
    match = re.search(pattern, text)
    return bool(match)

def is_title(text):
    """Is considered title every line that consists of a single capital letter or serie of underscore"""
    if re.fullmatch(r"[A-Z]|_+", text.strip()):
        return True
    else :
        return False
    
def skip_titles(lines, next_index):
    """Return the next sentence that is not a title or something like "1." """
    next_line = lines[next_index]

    while is_title(next_line) and next_index + 1 < len(lines) :
        next_index += 1
        next_line = lines[next_index]
      
    # avoid returning 1.  
    if re.match(r"^\d+\.", next_line.strip()):
        return sent_segmentation(next_line, "pySBD")[1]
    return sent_segmentation(next_line, "pySBD")[0]

def ends_with_syllabus(sentence):
    """True if the sentences ends with syllabus"""
    return bool(re.search(r'\bsyllabus\b$', sentence, re.IGNORECASE))
        
def sentence_after_syllabus(sentence):
    """Return the sentence following the word "syllabus" """
    match = re.search(r'\bsyllabus\b', sentence, re.IGNORECASE)
    if match:
        return sentence[match.end():].strip()
    else:
        return sentence

def select_query(text):
    """Return a sentence relevent for BM25 query"""
    query = ""
    
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()         
        # if there is a date like [Month day, year], we take the next line if it is not a title
        if contains_date_format(line) and i + 2 < len(lines):
            return sentence_after_syllabus(skip_titles(lines, i+2))
        
        # if the sentence contains "No.", we return the next sentence by skipping "Per Curiam" and "Syllabus"
        if "No." in line and i + 1 < len(lines):
            if "Per Curiam" not in lines[i + 1].strip():
                return sentence_after_syllabus(skip_titles(lines, i+1))
            elif i + 2 < len(lines):
                return sentence_after_syllabus(skip_titles(lines, i+2))
        
        # to not have the "Syllabus" conditions have the priority, store the query in a variable    
        # if the line is "Syllabus", the query will be the next line's first sentence
        if ("Syllabus" == line.strip() or ends_with_syllabus(line)) and i + 1 < len(lines):
            query = lines[i + 1]
        # if the line contains "syllabus", the query will be the next sentence
        elif "syllabus" in line.lower():
            query = sentence_after_syllabus(line)

    if query != "":
        return sent_segmentation(query, "pySBD")[0]
    else:
        return "law and legal rights"

def rouge_evaluations(text, ref, short=True):
    """Return a dataframe for rouge scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(text, ref)

    if short:
        return {
            'rouge1': scores['rouge1'].recall,
            'rouge2': scores['rouge2'].recall,
            'rougeL': scores['rougeL'].recall,
        }
    else:
        return {
            'rouge1': scores['rouge1'],
            'rouge2': scores['rouge2'],
            'rougeL': scores['rougeL'],
        }

def bert_evaluation(summary, ref, short=True):
    """Return a dataframe for bert score"""
    # Temporarily set verbosity to ERROR to suppress warnings
    logging.set_verbosity_error()

    try:
        scorer = BERTScorer(lang="en")
        precision, recall, f1 = scorer.score([summary], [ref])
        
        if short:
            return f1.mean().item()
        else:
            return [precision.mean().item(), recall.mean().item(), f1.mean().item()]
    finally:
        logging.set_verbosity_warning()
        
CLEANR = re.compile('<.*?>') 
def cleanhtml(raw_html):
    """Remove html tags"""
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

def evaluations(text, ref, short=True):
    """Return the different metrics results \n
        if short, return only the recall of rouge and f1_score of bert_score  """
    ref = cleanhtml(ref)
    rouges = rouge_evaluations(text, ref, short)
    bert = bert_evaluation(text, ref, short)
    
    if short:
        data = {
            'rouge1': [rouges['rouge1']],
            'rouge2': [rouges['rouge2']],
            'rougeL': [rouges['rougeL']],
            'bert_score': [bert]
        }
        return pd.DataFrame(data)
    else:
        data = {
            'rouge1_P': [rouges['rouge1'].precision],
            'rouge1_R': [rouges['rouge1'].recall],
            'rouge1_F1': [rouges['rouge1'].fmeasure],
            
            'rouge2_P': [rouges['rouge2'].precision],
            'rouge2_R': [rouges['rouge2'].recall],
            'rouge2_F1': [rouges['rouge2'].fmeasure],
            
            'rougeL_P': [rouges['rougeL'].precision],
            'rougeL_R': [rouges['rougeL'].recall],
            'rougeL_F1': [rouges['rougeL'].fmeasure],
            
            'bert_score_P': [bert[0]],
            'bert_score_R': [bert[1]],
            'bert_score_F1': [bert[2]]
        }
        
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
    '''extracts is a list of fragments/sentences that will be highlighted in full_text, which is an HTML.'''

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
    '''The extrats is a list of fragments/sentences that will be highlighted in full_text, which is a text.'''

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

def highlight_min_max(df, only_f1=True):
    """Highlight for results"""
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    # Appliquer le style pour les colonnes 'Precision', 'Recall', 'F1-Score'
    if only_f1:
        columns = ['rouge1', 'rouge2', 'rougeL', 'bert_score'] 
    else:
        columns = ['rouge1_F1', 'rouge2_F1', 'rougeL_F1', 'bert_score_F1'] 
        
    for col in columns:
        # Top 3 maximums et minimums
        top_3_max = df[col].nlargest(3)
        top_3_min = df[col].nsmallest(3)

        # Appliquer le dégradé rouge pour les min
        for i in df.index:
            if df[col].iloc[i] in top_3_min.values:
                rank = top_3_min.rank()[top_3_min == df[col].iloc[i]].values[0]
                alpha = 1 - (rank - 1) / 3  
                styles.loc[i, col]= f'background-color: rgba(200, 50, 50, {alpha});'

        # Appliquer le dégradé vert pour les max
        for i in df.index:
            if df[col].iloc[i] in top_3_max.values:
                rank = top_3_max.rank(ascending=False)[top_3_max == df[col].iloc[i]].values[0]
                alpha = 1 - (rank - 1) / 3
                styles.loc[i, col]= f'background-color: rgba(50, 200, 50, {alpha});'

    # Pour la colonne 'Execution time', inverser les couleurs (max en rouge, min en vert)
    col = 'Execution time'
    top_3_max = df[col].nlargest(3)
    top_3_min = df[col].nsmallest(3)

    # Appliquer le dégradé vert pour les min
    for i in df.index:
        if df[col].iloc[i] in top_3_min.values:
            rank = top_3_min.rank()[top_3_min == df[col].iloc[i]].values[0]
            alpha = 1 - (rank - 1) / 3 
            styles.loc[i, col]= f'background-color: rgba(50, 200, 50, {alpha});'

    # Appliquer le dégradé rouge pour les max
    for i in df.index:
        if df[col].iloc[i] in top_3_max.values:
            rank = top_3_max.rank(ascending=False)[top_3_max == df[col].iloc[i]].values[0]
            alpha = 1 - (rank - 1) / 3 
            styles.loc[i, col]= f'background-color: rgba(200, 50, 50, {alpha});'
            
    return styles