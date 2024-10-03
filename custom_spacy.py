import spacy
from spacy.language import Language
import re    
      
def custom_segmentation(doc):
    to_end_after_bracket = False
    for i, token in enumerate(doc[:-1]):

        # si mot qui créer une phrase
        if token.text == "Id." or token.text == "See":
            token.is_sent_start = True

        if token.text == "Ibid.":
            token.is_sent_start = True
            if i < len(doc) - 2 :
                doc[i + 1].is_sent_start = True

    
        # si point
        if token.text == ".":
            token.is_sent_start = False

            if i < len(doc) - 2:
                # Les parenthèses / crochets font partie de la phrase précédente
                if doc[i + 1].text in ["[", "("] or (i < len(doc) - 3 and doc[i + 2].text in ["[", "("]):
                    doc[i + 1].is_sent_start = False
                    to_end_after_bracket = True
                # Le point est suivi d'une minuscule   
                elif re.match(r'^[a-z]', doc[i + 1].text):
                    doc[i + 1].is_sent_start = False

                # si ressemble au point d'un sigle
                elif i > 0 and re.match(r'^[A-Z][A-Za-z]{0,3}$|^[a-z]{1,2}$', doc[i - 1].text):
                    doc[i + 1].is_sent_start = False

                # si dernier point des points de suspension    
                elif i > 2 and doc[i - 1].text == "." and doc[i - 2].text == ".":
                    doc[i + 1].is_sent_start = False

            continue

        # si fermer parenthèse ou crochet et to_end_after_bracket est vrai
        if token.text in ["]", ")"] and to_end_after_bracket:
            if i < len(doc) - 2:
                # si suivi de parenthèses ou crochets, la phrase continue, sinon fin de phrase
                if doc[i + 1].text not in ["(", "["]:
                    doc[i + 1].is_sent_start = True
                    to_end_after_bracket = False
                # si suivi ponctuation, la phrase continue
                elif doc[i + 1].text in [".", ";", ":"]:
                    to_end_after_bracket = False
            continue
        # si est entre parenthèse, continue la phrase    

        if to_end_after_bracket and i < len(doc) - 2:
            doc[i + 1].is_sent_start = False
            continue

        # si virgule
        if token.text == "," and i < len(doc) - 2:
            token.is_sent_start = False
            doc[i + 1].is_sent_start = False
            continue

        # si ponctuation, la ponctuation fait partie de la phrase précédente
        if token.text in [";", ":"]:
            token.is_sent_start = False
            if i < len(doc) - 2:
                # si suivie d'une minuscule, continue la phrase
                if re.match(r'^[a-z]', doc[i + 1].text):
                    doc[i + 1].is_sent_start = False
                elif doc[i + 1].text == "\"":
                    doc[i + 1].is_sent_start = True
                    if i < len(doc) - 3:
                        doc[i + 2].is_sent_start = False
                else:
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