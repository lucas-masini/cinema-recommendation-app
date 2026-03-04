# 🎬 CineOuest — Système de recommandation de films

## 🌐 Démo de l'application

L'application est disponible en ligne via Streamlit Cloud :

🔗 **Accéder à l'application :**  

https://cinema-recommendation-app.streamlit.app

## 📌 Contexte du projet

Dans le cadre d’un projet de **Data Analysis / Machine Learning**, notre équipe a été missionnée par un cinéma fictif nommé **CineOuest**, situé dans la région **Pays de la Loire**, afin de développer un service numérique pour accompagner son activité.

L’objectif est de proposer une **application permettant aux spectateurs de découvrir de nouveaux films grâce à un système de recommandation**, similaire à ceux utilisés par les plateformes de streaming.

L'application permet également de :

- consulter différentes **statistiques sur les films**
- explorer une **base de données enrichie**
- découvrir **de nouvelles recommandations personnalisées**

---

# 🎯 Objectifs du projet

Le projet repose sur trois objectifs principaux :

- 📊 **Analyser les données du marché du cinéma**
- 🎥 **Exploiter des bases de données publiques sur les films**
- 🤖 **Développer un système de recommandation basé sur le Machine Learning**

Le résultat final est une **application interactive développée avec Streamlit** permettant :

- 🔎 de rechercher un film
- 📖 d'obtenir des informations détaillées
- 🎯 de découvrir des films similaires

---

# 📚 Sources de données

Le projet s'appuie principalement sur deux sources de données.

## IMDb Datasets

Base de données publique contenant des informations sur :

- les films  
- les acteurs  
- les réalisateurs  
- les notes des utilisateurs  
- les genres  

🔗 https://datasets.imdbws.com/

---

## TMDB API (The Movie Database)

L'API **TMDB** a été utilisée afin d'enrichir les données avec :

- 🖼 les **affiches des films**
- 📝 les **résumés**
- 📊 certaines **métadonnées supplémentaires**

🔗 https://www.themoviedb.org/documentation/api

---

# 🧹 Préparation des données

Les datasets **IMDb** contiennent **plusieurs millions de films et d’acteurs**, ce qui nécessite un important travail de préparation avant l'analyse.

Plusieurs étapes ont été réalisées :

- sélection des **tables pertinentes**
- **nettoyage des données**
- **jointure entre les différentes sources**
- enrichissement avec les données **TMDB**
- **filtrage des données** afin d'obtenir une base exploitable pour l'analyse et le modèle de recommandation

Ce travail a permis de construire une base finale optimisée :


imdb_final.csv


Cette base contient notamment :

- 🎬 titre du film  
- 📅 année de sortie  
- 🏷 genres  
- 🎭 acteurs principaux  
- 🎥 réalisateurs  
- ⭐ note IMDb  
- 👍 nombre de votes  
- 📝 synopsis  
- 🖼 poster du film  

---

# 🤖 Système de recommandation

Le moteur de recommandation repose sur une approche **Content-Based Filtering**.

Le principe consiste à recommander des films similaires à un film donné en analysant leurs caractéristiques.

## Étapes du modèle

1️⃣ Transformation des données textuelles en vecteurs avec **TF-IDF**

2️⃣ Création d'une matrice de caractéristiques combinant :

- synopsis
- genres
- acteurs
- réalisateurs

3️⃣ Calcul de similarité entre films avec l’algorithme :


K-Nearest Neighbors (KNN)


Le système retourne ensuite une **liste de films similaires**.

---

# 💻 Application Streamlit

Une application interactive a été développée avec **Streamlit** afin de permettre aux utilisateurs de tester le système de recommandation.

## Fonctionnalités principales

### 🏠 Accueil

- affichage d’un **carrousel d’affiches de films**

### 🔎 Recherche de film

- recherche par **titre**
- affichage des **informations du film**
- suggestions de **films similaires**

### 🎛 Recherche par filtres

Possibilité de filtrer les films par :

- acteur
- réalisateur
- genre
- période de sortie

### 🎲 Film aléatoire

Proposition d’un **film choisi aléatoirement** avec recommandations associées.

### 📊 Exploration de la base

Affichage de statistiques comme :

- nombre de films
- note moyenne
- évolution des films par décennie

---

# 🛠 Technologies utilisées

- 🐍 **Python**
- 🐼 **Pandas**
- 🤖 **Scikit-learn**
- 📊 **TF-IDF Vectorizer**
- 🔎 **K-Nearest Neighbors**
- 🌐 **Streamlit**
- 📈 **Matplotlib**
- 🎬 **IMDb datasets**
- 🖼 **TMDB API**

---

# 📁 Structure du projet

```
cinema_recommandation-app
│
├── app
│ └── app_streamlit.py
│
├── images
│ ├── bandeau.png
│ └── salle_jaune_sombre.png
│ └── LOGO_DROITE.png
│
├── requirements.txt
├── imdb_final.csv
│
└── README.md
```

---

# 🚀 Perspectives d'amélioration

Plusieurs améliorations pourraient être apportées :

- amélioration du **moteur de recommandation**
- ajout d’un **filtrage collaboratif**
- ajout de **préférences utilisateurs**

---

# 🎓 Compétences développées

Ce projet m'a permis de développer des compétences en :

- préparation et **nettoyage de données**
- exploitation d’**API**
- **Machine Learning appliqué aux systèmes de recommandation**
- développement d’**applications interactives avec Streamlit**
- **visualisation et exploration de données**

---

# 👨‍💻 Contribution personnelle

Ce projet a été réalisé en équipe dans le cadre d'un projet de formation.

Mes principales contributions ont été :

- participation à la **préparation et au filtrage des données** issues des bases IMDb et TMDB
- contribution à la **création de la base de données finale (`imdb_final.csv`)**
- participation au développement de **l'application Streamlit**
- mise en place et tests de l'interface utilisateur
- participation à la **présentation du projet (PowerPoint)**
