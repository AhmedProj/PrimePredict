# PrimePredict

## Vue d'Ensemble
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
1. **models/ :** Contient les fichiers des modèles entraînés et les artéfacts de sérialisation.
2. **notebooks/ :** Cahiers Jupyter avec analyse exploratoire des données, entraînement des modèles et expérimentations d'évaluation.
3. **src/ :**  Code source pour le projet incluant le prétraitement des données, les définitions des modèles, les scripts d'entraînement et les configurations de déploiement.
4. **README.md** : Fournit une vue d'ensemble et des instructions pour le dépôt.


## Pour Commencer
Clonez le dépôt et installez les dépendances nécessaires avec les commandes suivantes :
```bash
git clone https://github.com/AhmedProj/PrimePredict.git
cd PrimePredict
pip install -r requirements.txt
