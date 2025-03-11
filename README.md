# Sentiment Analysis Pipeline

# Introduction
Projet : Construction d'un pipeline collaboratif d'analyse de sentiment en utilisant un modèle BERT.

Objectif : Analyser les sentiments à partir de textes en utilisant un modèle de traitement du langage naturel (NLP) basé sur BERT.

Collaboration : Travail en binôme avec Hergi DIANGUE et Bally-Stone.

Structure du Projet
Le projet est divisé en trois composants principaux :

Extraction des Données :

Responsable : Hergi DIANGUE.

Tâches :

Charger les données brutes depuis un fichier CSV.

Vérifier la présence des colonnes essentielles (content, score).

Gérer les erreurs (fichiers manquants, formats incorrects).

Tests : Vérification du chargement des données et des colonnes requises.

Traitement des Données :

Responsables : Hergi DIANGUE  et Bally-Stone .

Tâches :

Nettoyer le texte (suppression des caractères spéciaux, normalisation, etc.).

Tokeniser le texte avec le tokenizer BERT (bert-base-uncased).

Diviser les données en ensembles d'entraînement et de validation.

Tests : Vérification du nettoyage du texte et de la tokenisation.

Entraînement et Inférence du Modèle :

Responsable : Bally-Stone.

Tâches :

Charger un modèle BERT pré-entraîné pour la classification de séquences.

Affiner le modèle sur les données de sentiment.

Créer un script d'inférence pour prédire le sentiment de nouveaux textes.

Tests : Vérification du modèle et de l'inférence.

Collaboration et Flux de Travail Git
Branches :

feature-data-extraction pour l'extraction des données (Hergi DIANGUE).

feature-data-processing pour le traitement des données (Hergi DIANGUE et Bally-Stone).

feature-model-training pour l'entraînement du modèle (Bally-Stone).

Pull Requests : Chaque branche est fusionnée via une Pull Request avec revue de code obligatoire.


Livrables
Dépôt GitHub Public :

Structure du projet avec preuves de collaboration (branches, Pull Requests, historique des commits).

Documentation :

README.md avec instructions de configuration, exemples d'utilisation et description des composants.

Rapport de Projet :

Aperçu de l'approche, division du travail, défis rencontrés et améliorations futures.

Ressources
Kaggle : Sentiment Analysis using BERT

GitHub CheatSheet : Git Cheat Sheet

# Conclusion
Impact : Ce projet permet de comprendre comment construire un pipeline complet d'analyse de sentiment en utilisant des modèles de NLP modernes comme BERT.

Collaboration : La division des tâches et la revue de code ont permis une meilleure compréhension des différentes étapes du pipeline.

Améliorations Futures : Intégration de modèles plus avancés, amélioration de la gestion des erreurs, et extension à d'autres langues.
