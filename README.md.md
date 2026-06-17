

# Projet de Fin de Module — Deep Learning

## EMSI Casablanca — Filière Informatique, 4IIR-IAD

### Année universitaire 2025–2026

## Conception, implémentation, comparaison et analyse critique de modèles de Deep Learning pour données tabulaires, images et séquences

**Auteur : Safaa Hassani**

**Encadrant : Mr. Batal**

---

# Description

Ce projet de fin de module a pour objectif d'étudier, implémenter, comparer et analyser plusieurs architectures majeures du Deep Learning à l'aide du framework PyTorch.

Trois familles de modèles ont été développées et évaluées sur des jeux de données réels représentant différents types de données : tabulaires, images et séquences textuelles.

| Partie | Architecture               | Dataset                 | Tâche                                               |
| ------ | -------------------------- | ----------------------- | --------------------------------------------------- |
| I      | MLP                        | Breast Cancer Wisconsin            | Accuracy, Precision, Recall, F1-score, Matrice de confusion   |
| II     | CNN (LeNet)                | CIFAR-10                | Classification d'images couleur                     |
| III    | RNN / LSTM / GRU / Seq2Seq | Corpus Anglais–Espagnol | Modélisation de séquences et traduction automatique |

---

# Structure du Projet

```text
.
├── Partie1_MLP.ipynb
├── Partie2_CNN_CIFAR10.ipynb
├── Partie3_RNN_EN_ES.ipynb
├── Rapport_DeepLearning.pdf
├── README.md
└── data/
```

---

# Environnement d’Exécution

Les notebooks ont été développés sous :

* Python 3.10+
* PyTorch
* Google Colab
* Jupyter Notebook

## Dépendances Principales

```python
torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
```

Installation :

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn nltk
```

---

# Partie I — Perceptron Multicouche (MLP)

## Dataset : Breast Cancer Wisconsin

Le dataset Breast Cancer Wisconsin est un jeu de données médical largement utilisé pour la détection du cancer du sein.

### Caractéristiques

* 569 observations
* 30 caractéristiques numériques décrivant les cellules tumorales
* Aucune valeur manquante
* Classification binaire :
  * Bénin (Benign)
  * Malin (Malignant)

## Travaux Réalisés

* Analyse exploratoire des données
* Prétraitement et normalisation avec `StandardScaler`
* Prévention du data leakage
* Séparation Train / Validation / Test
* Implémentation d'un MLP avec `nn.Sequential`
* Implémentation d'un MLP via une classe personnalisée `nn.Module`
* Comparaison de plusieurs stratégies d'initialisation :
  * Initialisation Gaussienne
  * Initialisation Constante
  * Initialisation Xavier
* Mise en place de l'Early Stopping
* Régularisation par Dropout et Weight Decay (L2)
* Sauvegarde et rechargement du modèle
* Comparaison avec des modèles classiques :
  * Régression Logistique
  * Forêt Aléatoire
* Exécution sur CPU/GPU

## Architecture du Réseau

```text
30 → 64 → ReLU → Dropout(0.3) → 2
```

## Concepts Supplémentaires Étudiés

* Xavier Initialization
* Early Stopping
* Dropout
* Weight Decay (L2 Regularization)
* Comparaison avec des modèles de Machine Learning classiques
* Sauvegarde du meilleur modèle avec `state_dict`

## Métriques d'Évaluation

* Accuracy
* Precision
* Recall
* F1-score
* Matrice de confusion

---

# Partie II — Réseaux de Neurones Convolutionnels (CNN)

## Dataset : CIFAR-10

Le dataset CIFAR-10 est composé d'images couleur réparties en 10 classes.

### Caractéristiques

* 50 000 images d'entraînement
* 10 000 images de test
* Images RGB
* Taille : 32 × 32 pixels
* 10 catégories d'objets

## Travaux Réalisés

### Étude Théorique

* Corrélation croisée 2D
* Padding
* Stride
* Max Pooling
* Average Pooling
* Convolution 1×1

### Implémentations Manuelles

* Corrélation croisée 2D
* Max Pooling
* Average Pooling

### Comparaison avec PyTorch

Validation des implémentations manuelles avec les couches natives de PyTorch.

### Architecture CNN

Implémentation d'une version adaptée du réseau LeNet pour images RGB.

### Entraînement

* Optimiseur : Adam
* Fonction de perte : CrossEntropyLoss
* Nombre d'époques : 8

## Métriques d'Évaluation

* Accuracy
* Classification Report
* Matrice de confusion

---

# Partie III — RNN, LSTM, GRU et Seq2Seq

## Dataset : Corpus Anglais → Espagnol

Cette partie porte sur la modélisation de séquences et la traduction automatique.

### Prétraitement

* Tokenisation
* Construction du vocabulaire
* Padding
* Mini-batchs
* Encodage des séquences

## Travaux Réalisés

### Modèles Réccurents

* RNN Simple
* LSTM
* GRU

### Analyse Expérimentale

Comparaison des architectures selon :

* Stabilité
* Temps d'entraînement
* Capacité de mémorisation
* Performance globale

### Techniques d'Apprentissage

* BPTT (Backpropagation Through Time)
* Gradient Clipping

### Système Seq2Seq

Implémentation complète :

* Encodeur
* Décodeur
* Teacher Forcing

### Décodage

* Greedy Decoding
* Beam Search

## Métriques d'Évaluation

* Perplexité
* BLEU Score

---

# Résultats Généraux

| Partie  | Dataset            | Métriques                                             |
| ------- | ------------------ | ----------------------------------------------------- |
| MLP     | Wine Quality       | Accuracy, Precision, Recall, F1-score                 |
| CNN     | CIFAR-10           | Accuracy, Classification Report, Matrice de confusion |
| Seq2Seq | Anglais → Espagnol | Perplexité, BLEU Score                                |

Les résultats obtenus montrent que :

* Les MLP sont particulièrement adaptés aux données tabulaires structurées.
* Les CNN exploitent efficacement les relations spatiales des images couleur grâce au partage des poids et aux filtres convolutifs.
* Les architectures récurrentes permettent de modéliser les dépendances temporelles et séquentielles.
* Les modèles Seq2Seq offrent une solution efficace pour les tâches de traduction automatique.

---

# Conclusion

Ce projet a permis de mettre en œuvre les principales familles d'architectures étudiées dans le module de Deep Learning.

L'étude comparative met en évidence que chaque architecture répond à un type spécifique de données :

* **MLP** → données tabulaires
* **CNN** → données visuelles
* **RNN / LSTM / GRU / Seq2Seq** → données séquentielles

Les résultats obtenus confirment l'importance du choix de l'architecture en fonction de la nature du problème étudié et démontrent l'efficacité des outils proposés par PyTorch pour le développement de modèles de Deep Learning.

---

# Licence

Projet académique réalisé dans le cadre du module **Deep Learning**.

**EMSI Casablanca — Année Universitaire 2025–2026**

Travail individuel.
