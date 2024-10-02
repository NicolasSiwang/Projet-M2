# import spacy
# from spacy.language import Language
# import re

# def custom_segmentation(doc):
#     """
#     Fonction de segmentation personnalisée qui modifie le comportement par défaut de la segmentation des phrases de spaCy.
    
#     Parameters:
#     doc (spacy.tokens.Doc): Le document spaCy à traiter.

#     Returns:
#     spacy.tokens.Doc: Le document avec des modifications dans les marqueurs de début de phrase.
#     """
#     to_end_after_bracket = False  # Indicateur pour gérer les parenthèses/crochets
#     for i, token in enumerate(doc[:-1]):
        
#         # Si le mot crée une nouvelle phrase
#         if token.text == "Id." or token.text == "See":
#             token.is_sent_start = True
        
#         # Si le mot est "Ibid."
#         if token.text == "Ibid.":
#             token.is_sent_start = True
#             if i < len(doc) - 2:
#                 doc[i + 1].is_sent_start = True

#         # Si le token est un point
#         if token.text == ".":
#             token.is_sent_start = False
                
#             # Les parenthèses/crochets font partie de la phrase précédente
#             if doc[token.i + 1].text in ["[", "("] or doc[token.i + 2].text in ["[", "("]:
#                 doc[token.i + 1].is_sent_start = False
#                 to_end_after_bracket = True
#             # Si le point est suivi d'une minuscule
#             elif re.match(r'^[a-z]', doc[i + 1].text):
#                 doc[i + 1].is_sent_start = False
            
#             # Si ressemble à un point d'un sigle
#             elif (i > 0 and re.match(r'^[A-Z][A-Za-z]{0,3}$|^[a-z]{1,2}$', doc[i-1].text) and i < len(doc) - 2):
#                 doc[i + 1].is_sent_start = False
            
#             # Si dernier point de points de suspension
#             elif (i > 2 and doc[i - 1].text == "." and doc[i - 2].text == ".") and i < len(doc) - 2:
#                 doc[i + 1].is_sent_start = False
                
#             continue
        
#         # Si on rencontre une parenthèse ou un crochet et que to_end_after_bracket est vrai
#         if (token.text == "]" or token.text == ")") and to_end_after_bracket:
#             # Si suivi de parenthèses ou crochets, la phrase continue, sinon fin de phrase
#             if (doc[i + 1].text != "(" or doc[i + 1].text != "["):
#                 doc[i + 1].is_sent_start = True
#                 to_end_after_bracket = False
#             # Si suivi de ponctuation, la phrase continue
#             elif doc[i + 1].text in [".", ";"] or token.text == ":":
#                 to_end_after_bracket = False
#             continue
        
#         # Si on est dans un contexte où to_end_after_bracket est vrai
#         if to_end_after_bracket:
#             doc[i + 1].is_sent_start = False
#             continue
        
#         # Si on rencontre une virgule
#         if token.text == "," and i < len(doc) - 2:
#             token.is_sent_start = False
#             doc[i + 1].is_sent_start = False
#             continue
            
#         # Si ponctuation qui fait partie de la phrase précédente
#         if token.text in [";", ":"]:
#             token.is_sent_start = False
#             # Si suivi d'une minuscule, ne marque pas la fin de la phrase
#             if (i < len(doc) - 2 and re.match(r'^[a-z]', doc[i + 1].text)):
#                 doc[i + 1].is_sent_start = False
#             elif (i < len(doc) - 2 and doc[i + 1].text == "\""):
#                 doc[i + 1].is_sent_start = True
#                 if i < len(doc) - 3:
#                     doc[i + 2].is_sent_start = False
#             else:
#                 doc[i + 1].is_sent_start = True
#             continue
            
#     return doc

# @Language.component("custom_segmentation")
# def custom_segmentation_component(doc):
#     """
#     Composant de segmentation personnalisé pour spaCy.
    
#     Parameters:
#     doc (spacy.tokens.Doc): Le document spaCy à traiter.

#     Returns:
#     spacy.tokens.Doc: Le document avec des modifications dans les marqueurs de début de phrase.
#     """
#     return custom_segmentation(doc)

# def custom_spacy_model():
#     """
#     Fonction pour créer un modèle spaCy personnalisé avec une segmentation de phrases.
    
#     Returns:
#     spacy.language.Language: Le modèle spaCy personnalisé.
#     """
#     nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Chargement du modèle spaCy
#     nlp.add_pipe("sentencizer")  # Ajout d'un composant de segmentation de phrases
#     nlp.add_pipe("custom_segmentation", before="tagger")  # Ajout du composant personnalisé avant le tagger

#     nlp.to_disk("models/custom_spacy_model")  # Enregistrement du modèle personnalisé
    
#     return nlp

import spacy
from spacy.language import Language
import re    
      
def custom_segmentation(doc):
    to_end_after_bracket = False
    for i, token in enumerate(doc[:-1]):

        # si mot qui créer une phrase
        if token.text == "Id." or token.text == "See":
            token.is_sent_start = True

        if token.text == "Ibid." :
            token.is_sent_start = True
            if i < len(doc) - 2 :
                doc[i + 1].is_sent_start = True

        
        # si point
        if token.text == ".":
            token.is_sent_start = False               
            
            # Les parenthèses / crochets font partie de la phrase précédente
            if doc[token.i + 1].text == "[" or doc[token.i + 1].text == "(" or doc[token.i + 2].text == "[" or doc[token.i + 2].text == "(":
                doc[token.i + 1].is_sent_start = False
                to_end_after_bracket = True
            # Le point est suivi d'une minuscule   
            elif re.match(r'^[a-z]', doc[i + 1].text):
                doc[i + 1].is_sent_start = False

            # si ressemble au point d'un sigle
            elif (i > 0 and re.match(r'^[A-Z][A-Za-z]{0,3}$|^[a-z]{1,2}$', doc[i-1].text) and i < len(doc) - 2):
                doc[i + 1].is_sent_start = False

            # si dernier point des points de suspension    
            elif (i > 2 and doc[i - 1].text == "." and doc[i - 2].text == ".")  and i < len(doc) - 2:
                doc[i + 1].is_sent_start = False

            continue

        # si fermer parenthèse ou crochet et to_end_after_bracket est vrai
        if (token.text == "]" or token.text == ")") and to_end_after_bracket:
            # si suivi de parenthèses ou crochets, la phrase continue, sinon fin de phrase
            if (doc[i + 1].text != "(" or doc[i + 1].text != "["):
                doc[i + 1].is_sent_start = True
                to_end_after_bracket = False
            # si suivi ponctuation, la phrase continue
            elif (doc[i+1].text == "." or doc[i+1].text == ";" or token.text == ":"):
                to_end_after_bracket = False
            continue

        # si est entre parenthèse, continue la phrase    
        if to_end_after_bracket :
            doc[i + 1].is_sent_start = False
            continue

        # si virgule, la virgule fait partie de la phrase précédente et pas de fin de phrase
        if token.text == "," and i < len(doc) - 2 :
            token.is_sent_start = False
            doc[i + 1].is_sent_start = False
            continue

        # si ponctuation, la ponctuation fait partie de la phrase précédente
        if token.text == ";" or token.text == ":":
            token.is_sent_start = False
            # si suivie d'une minuscule, ne marque pas la fin de la phrase, sinon oui
            if (i < len(doc) - 2 and re.match(r'^[a-z]', doc[i + 1].text)):
                doc[i + 1].is_sent_start = False
            elif (i < len(doc) - 2 and doc[i + 1].text == "\""):
                doc[i + 1].is_sent_start = True
                if i < len(doc) - 3:
                    doc[i + 2].is_sent_start = False
            else :
                doc[i + 1].is_sent_start = True
            continue
    return doc

@Language.component("custom_segmentation")
def custom_segmentation_component(doc):
    return custom_segmentation(doc)

def custom_spacy_model():
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer") 
    nlp.add_pipe("custom_segmentation", before="tagger")

    nlp.to_disk("models/custom_spacy_model")

    return nlp