# PrimePredict


```
export MLFLOW_MODEL_FREQ_NAME="model_freq"
export MLFLOW_MODEL_FREQ_VERSION=1 
export MLFLOW_MODEL_REG_NAME="model_reg"
export MLFLOW_MODEL_REG_VERSION=1
```

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/username/PrimePredict/CI)
![Codecov](https://codecov.io/gh/username/PrimePredict/branch/main/graph/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)



## Description du Projet
Ce dépôt contient le développement d'un modèle de machine learning avancé destiné à prédire le total des paiements de sinistres d'assurance. Notre méthode est conçue pour faciliter le calcul des primes annuelles matérielles pour un jeu de données comprenant 36 311 contrats d'assurance pour l'année 2011. Le projet englobe non seulement le développement du modèle mais aussi son déploiement pour une utilisation en production.


## Objectif
L'objectif principal de ce projet est de calculer les primes annuelles matérielles pour l'ensemble de données fourni. Pour atteindre cet objectif, nous adoptons une approche de modélisation en deux étapes qui prédit à la fois le coût des dommages matériels et la fréquence des incidents matériels.

## Modèles
Nous développerons et évaluerons les modèles suivants :

1. **Modèle d'Estimation des Coûts Matériels :** Ce modèle sera formé pour estimer les coûts associés aux dommages matériels résultant d'incidents.
2. **Modèle de Prédiction de la Fréquence des Incidents Matériels :** Ce modèle vise à prédire la fréquence des incidents causant des dommages matériels.

## Calcul de la Prime
La Prime Prédite est calculée en utilisant la formule suivante :

`Prime Prédite = Fréquence Prédite * Coût Moyen Prédit`

## Structure du Dépôt

- **.github/workflows/** : Contient les fichiers de configuration pour les workflows GitHub Actions utilisés pour l'intégration continue et le déploiement continu.
- **Streamlit/** : Dossiers des scripts Streamlit pour créer des applications web interactives permettant de visualiser les résultats des prédictions.
- **app/** : Code source de l'application principale, y compris les améliorations pour le score pylint.
- **argocd/** : Fichiers de configuration pour Argo CD, utilisé pour le déploiement déclaratif des applications dans Kubernetes.
- **kubernetes/** : Contient les fichiers de configuration Kubernetes, y compris les fichiers ingress pour la gestion du trafic réseau.
- **models/** : Dossiers pour les modèles entraînés, y compris les fichiers pour la gestion et le suivi des modèles.
- **notebooks/** : Jupyter notebooks pour l'analyse exploratoire des données et les tests de validation des modèles.
- **src/** : Code source du projet contenant les scripts de prétraitement des données, les définitions des modèles, et plus.
- **unit_test/** : Tests unitaires pour vérifier la fonctionnalité des modules de code.
- **Dockerfile** : Fichier Docker pour construire les images des conteneurs utilisés dans le projet.
- **README.md** : Document principal fournissant une vue d'ensemble et des instructions détaillées pour le dépôt.
- **requirements.txt** : Liste toutes les dépendances Python nécessaires pour le projet.


## Pour Commencer

Pour utiliser ce projet, suivez ces étapes :

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/AhmedProj/PrimePredict.git
   cd PrimePredict
## Pour Commencer

Pour utiliser ce projet, suivez ces étapes :

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/AhmedProj/PrimePredict.git
   cd PrimePredict
2. **Configurer un environnement virtuel :**
   python -m venv venv
source venv/bin/activate  # Sur Windows utilisez `venv\Scripts\activate`
3. ** Installer les dépendances :**
   pip install -r requirements.txt
4. **Lancer l'application :**

## Tests

Pour exécuter les tests unitaires, utilisez la commande suivante :
```bash
python -m unittest discover -s unit_test



