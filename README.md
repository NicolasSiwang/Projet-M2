# README

## Résumé du projet

Ce projet vise à développer un système hybride de résumé automatique pour des documents juridiques, tels que les décisions de la Cour suprême des États-Unis. En combinant des approches extractives et abstractive, notre solution cherche à générer des résumés à la fois concis, informatifs et lisibles, tout en capturant l’essence des textes juridiques.

### Méthodologie
1. **Récupération et pondération** :
   - Utilisation du schéma de pondération BM25 pour identifier les documents les plus pertinents.
2. **Clustering sémantique** :
   - Regroupement des phrases similaires en utilisant des modèles basés sur BERT.
3. **Résumé hybride** :
   - **Composant extractif** : Sélection des phrases clés en fonction de leur pertinence.
   - **Composant abstractive** : Reformulation et amélioration de la lisibilité grâce à des modèles comme BART ou Legal-Pegasus.

---

## Prérequis

- **Langage** : Python (3.8 ou supérieur)
- **Bibliothèques principales** :
  - `spacy` (pour le traitement de texte avancé)
  - `pandas` (pour la manipulation des données tabulaires)
  - `rank_bm25` (pour la pondération BM25)
  - `transformers` (pour les modèles de NLP comme BERT et BART)
  - `torch` (pour l'exécution de modèles de deep learning)
- **Environnement d'exécution recommandé** : GPU (pour l'exécution efficace des modèles de deep learning)

---

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votre-projet/legal-summarization.git
   cd legal-summarization

## Utilisation

1. **Préparation des données** :
   Placez les documents juridiques dans un dossier nommé `data/` au format texte ou JSON.

2. **Exécution du pipeline** :
   Lancez le script principal :
   ```bash
   python main.py --input data/ --output summaries/
   ```

--input : Chemin du dossier contenant les documents à résumer.
--output : Chemin du dossier où seront enregistrés les résumés générés.
Options de configuration :

## Licence
