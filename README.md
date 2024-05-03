# PrimePredict


## Description du Proje
ICI  Ce dépôt contient le développement d'un modèle de machine learning avancé destiné à prédire le total des paiements de sinistres d'assurance. Notre méthode est conçue pour faciliter le calcul des primes annuelles matérielles pour un jeu de données comprenant 36 311 contrats d'assurance pour l'année 2011.

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
- **argo_workflow/** : Dossiers des configuration yaml permattant de lancer des entraînements en parallèle.
- **app/** : Code source de l'application principale, y compris les améliorations pour le score pylint.
- **argocd/** : Fichiers de configuration pour Argo CD, utilisé pour le déploiement déclaratif des applications dans Kubernetes.
- **kubernetes/** : Contient les fichiers de configuration Kubernetes, y compris les fichiers ingress pour la gestion du trafic réseau.
- **models/** : Dossiers pour les modèles entraînés, y compris les fichiers pour la gestion et le suivi des modèles.
- **notebooks/** : Jupyter notebooks pour l'analyse exploratoire des données et les tests de validation des modèles.
- **src/** : Code source du projet contenant les scripts de prétraitement des données, les définitions des modèles, et plus.
- **unit_test/** : Tests unitaires pour vérifier la fonctionnalité des modules de code.
- **Dockerfile** : Fichier Docker pour construire les images des conteneurs utilisés dans le projet.
- **README.md** : Document principal fournissant une vue d'ensemble et des instructions détaillées pour le dépôt.
- **MLproject** : Fichier permettant de lancer des entraînements sur MLflow dans un format mlflow project. 
- **requirements.txt** : Liste toutes les dépendances Python nécessaires pour le projet.


## Pour Commencer

Pour utiliser ce projet, suivez ces étapes :

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/AhmedProj/PrimePredict.git
   cd PrimePredict
   pip install -r requirements.txt
## Pour Commencer

Pour utiliser ce projet, suivez ces étapes :

1. **Configurer déjà des variables d'environements:**
   ```bash

2. **Lancer un entraînement qui utilise le fichier MLproject:**
   

4. ** Installer les dépendances :**
   pip install -r requirements.txt
5. **Lancer l'application :**

## Tests

Pour exécuter les tests unitaires, utilisez la commande suivante :
```bash
python -m unittest discover -s unit_test



