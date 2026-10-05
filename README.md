# Prédiction des maladies cardiaques par fouille de données

**Groupe 3** : Ridore Wenchel, Kandolo Herman Herman
**Cours** : Fouille de données
**Notebook** : `FDAnalyse.ipynb` (Google Colab)

---

## 1. Présentation

Les maladies cardiaques figurent parmi les premières causes de mortalité dans le monde. Ce projet étudie la possibilité de prédire la présence d'une maladie cardiaque à partir de mesures cliniques simples : âge, tension artérielle, cholestérol, résultats d'un électrocardiogramme d'effort, etc.

La démarche suit les étapes classiques d'un projet de fouille de données :

```
Données brutes -> Nettoyage -> Exploration -> Préparation -> Analyse non supervisée -> Modèles prédictifs -> Comparaison
```

La fouille de données consiste à extraire des connaissances exploitables d'un grand tableau de données. Elle commence toujours par la compréhension et le nettoyage des données, avant toute modélisation.

---

## 2. Données

| Élément | Description |
|---|---|
| Fichier | `heart excel.xlsx` (Google Drive, dossier `Fouille_de_données`) |
| Taille | 918 patients (lignes) et 12 variables (colonnes) |
| Variable cible | `HeartDisease` : 1 = malade (508 patients), 0 = sain (410 patients) |

### Dictionnaire des variables

| Variable | Type | Signification |
|---|---|---|
| `Age` | Numérique | Âge du patient (28 à 77 ans) |
| `Sex` | Catégorielle | M (homme) ou F (femme) |
| `ChestPainType` | Catégorielle | Type de douleur thoracique : ATA (angine atypique), NAP (douleur non angineuse), TA (angine typique), ASY (asymptomatique) |
| `RestingBP` | Numérique | Tension artérielle au repos (mm Hg) |
| `Cholesterol` | Numérique | Cholestérol sérique (mg/dL) |
| `FastingBS` | Binaire | Glycémie à jeun supérieure à 120 mg/dL (1 = oui) |
| `RestingECG` | Catégorielle | Résultat de l'ECG au repos (Normal, ST, LVH) |
| `MaxHR` | Numérique | Fréquence cardiaque maximale atteinte |
| `ExerciseAngina` | Binaire | Angine provoquée par l'effort (Y ou N) |
| `Oldpeak` | Numérique | Dépression du segment ST induite par l'effort |
| `ST_Slope` | Catégorielle | Pente du segment ST à l'effort (Up, Flat, Down) |
| `HeartDisease` | Binaire | Variable cible : 1 = maladie cardiaque |

---

## 3. Déroulement du notebook

### Étape 1 : Exploration et nettoyage

Les données sont chargées avec pandas, puis inspectées (dimensions, valeurs minimales et maximales, valeurs manquantes).

Anomalie détectée : 172 patients présentent un cholestérol égal à 0, valeur impossible sur le plan biologique. Ces zéros correspondent à des valeurs manquantes. Le même traitement est appliqué à `RestingBP` et `MaxHR`, qui contiennent également des zéros aberrants.

Traitement retenu : remplacement des zéros par la médiane de la colonne. La médiane est préférée à la moyenne car elle est peu sensible aux valeurs extrêmes.

### Étape 2 : Analyse descriptive

Pour chaque variable numérique, le notebook calcule la moyenne, la médiane, l'écart-type et le coefficient d'asymétrie (skewness). Cela permet d'identifier les distributions déséquilibrées : `Cholesterol` et `Oldpeak` présentent une longue queue vers les valeurs élevées.

### Étape 3 : Analyse bivariée avec la variable cible

Chaque variable est confrontée à `HeartDisease` à l'aide de graphiques et de tests statistiques.

| Variable | Outil | Constat |
|---|---|---|
| `Age` | Boxplot, diagramme en barres | Les patients malades sont plus âgés (corrélation d'environ 0,28) |
| `Sex` | Diagrammes en barres et circulaires | 63 % des hommes sont malades, contre 26 % des femmes |
| `ChestPainType` | Test du Chi² | Type ASY : environ 79 % de malades ; type ATA : environ 14 % |
| `RestingBP` | Test t de Student | Tension légèrement plus élevée chez les malades (134 contre 130 mm Hg, p = 0,0003) |
| `Cholesterol`, `MaxHR` | Courbes de densité (KDE) | Distributions différentes selon le groupe |
| `ExerciseAngina` | Test du Chi² | Lien marqué : 316 malades sur 371 patients présentant une angine d'effort |
| `Oldpeak`, `ST_Slope` | Graphiques et Chi² | Liens très significatifs (p < 0,0001) |

Le test du Chi² vérifie l'existence d'un lien entre deux variables catégorielles. Le test t compare la moyenne d'une variable numérique entre deux groupes. Dans les deux cas, une p-value inférieure à 0,05 indique que l'écart observé ne s'explique vraisemblablement pas par le hasard.

### Étape 4 : Préparation des données

Les algorithmes de modélisation exigent des variables numériques. Les transformations suivantes sont appliquées :

- Codage binaire de `Sex` (M = 1, F = 0).
- Encodage One-Hot de `ChestPainType`, `RestingECG`, `ExerciseAngina` et `ST_Slope` : chaque modalité devient une colonne 0/1. Le jeu final comporte 20 variables.
- Normalisation par `RobustScaler` : chaque variable est centrée sur sa médiane puis divisée par l'écart interquartile (IQR), ce qui limite l'influence des valeurs extrêmes.

### Étape 5 : Corrélations et ACP

- La matrice de corrélation (carte de chaleur) présente les relations linéaires entre toutes les variables.
- L'Analyse en Composantes Principales (ACP) résume les 20 variables en un nombre réduit d'axes. Six composantes expliquent environ 75 % de la variance (PC1 : 24 %, PC2 : 21 %).
- La projection des patients sur le plan PC1-PC2, colorée selon `HeartDisease`, montre une séparation correcte des deux groupes.

L'ACP remplace plusieurs variables corrélées par un petit nombre de nouvelles variables, en conservant le maximum d'information.

### Étape 6 : Classification non supervisée

La variable cible est ignorée et les patients sont regroupés selon leur ressemblance :

- Classification Ascendante Hiérarchique (CAH), représentée par un dendrogramme.
- K-means, avec choix du nombre de groupes par la méthode du coude (inertie intra-classe WCSS pour k allant de 1 à 10). Le notebook retient k = 3.

### Étape 7 : Modélisation prédictive

Les données sont séparées en 80 % pour l'entraînement et 20 % pour le test (184 patients), avec stratification pour conserver les proportions de chaque classe.

Les classes étant légèrement déséquilibrées, la méthode SMOTE est appliquée sur l'ensemble d'entraînement uniquement : elle génère des observations synthétiques de la classe minoritaire (406 observations par classe après rééquilibrage).

| Modèle | Entrée | Particularité |
|---|---|---|
| Réseau de neurones (Keras) | 20 variables | SMOTE, 100 époques maximum |
| Réseau de neurones (MLP, scikit-learn) | 6 composantes ACP | Architecture 256-128-64-32 |
| Régression logistique | 20 variables | SMOTE et seuil de décision optimisé |
| Régression logistique avec ACP | 6 composantes ACP | SMOTE et seuil de décision optimisé |

Un modèle de classification fournit une probabilité d'appartenance à la classe « malade ». Le seuil par défaut est de 0,5 ; il peut être ajusté afin de maximiser le F1-score.

---

## 4. Résultats

Performances mesurées sur les 184 patients de test :

| Modèle | Accuracy | F1 (classe malade) | AUC-ROC |
|---|---|---|---|
| Réseau de neurones (Keras) | 0,86 | 0,87 | non calculé |
| Régression logistique (seuil 0,49) | 0,90 | 0,91 | 0,925 |
| Régression logistique avec ACP (seuil 0,39) | 0,90 | 0,91 | 0,924 |

La régression logistique, plus simple, obtient des performances égales ou supérieures à celles du réseau de neurones. La réduction à 6 composantes par ACP ne dégrade pratiquement pas les résultats tout en divisant par plus de trois le nombre de variables.

Les modèles entraînés sont sauvegardés (fichiers `.pkl` et `.h5`) en vue d'une réutilisation.

---

## 5. Glossaire

| Terme | Définition |
|---|---|
| Imputation | Remplacement d'une valeur manquante par une valeur plausible (ici, la médiane) |
| Médiane | Valeur centrale : la moitié des observations lui est inférieure, l'autre moitié supérieure |
| IQR | Écart entre le 75e et le 25e percentile |
| One-Hot Encoding | Transformation d'une variable catégorielle en colonnes binaires |
| ACP (PCA) | Réduction du nombre de variables avec conservation de l'essentiel de l'information |
| K-means, CAH | Méthodes de regroupement automatique d'individus similaires |
| SMOTE | Génération d'observations synthétiques pour équilibrer les classes |
| Matrice de confusion | Tableau croisant prédictions et réalité (vrais et faux positifs et négatifs) |
| F1-score | Moyenne harmonique de la précision et du rappel |
| AUC-ROC | Capacité du modèle à distinguer les deux classes (1 = parfait, 0,5 = hasard) |

---

## 6. Reproduction

### Prérequis

- Un compte Google (le notebook est conçu pour Google Colab).
- Le fichier `heart excel.xlsx` placé dans `Mon Drive/Fouille_de_données/`.

### Bibliothèques

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn tensorflow openpyxl
```

### Exécution

1. Ouvrir `FDAnalyse.ipynb` dans Google Colab.
2. Exécuter la cellule de montage de Google Drive et autoriser l'accès.
3. Vérifier que la variable `chemin_fichier` pointe vers la copie du fichier Excel.
4. Exécuter les cellules dans l'ordre (Exécution, puis Tout exécuter).

Hors Colab, la cellule `drive.mount(...)` doit être supprimée et le chemin du fichier remplacé par un chemin local.

---

## 7. Limites et perspectives

- Jeu de données de taille limitée (918 patients, 184 en test) : les scores peuvent varier selon le découpage. Une validation croisée fournirait une estimation plus fiable.
- Le remplacement des zéros par la médiane est simple mais réduit la variabilité réelle ; une imputation par k plus proches voisins pourrait être envisagée.
- Les valeurs aberrantes ne sont traitées que par la normalisation robuste ; une analyse plus fine est possible.
- Le notebook contient des cellules dupliquées (chargement, nettoyage du cholestérol) et des avertissements `FutureWarning` de pandas ; un nettoyage améliorerait sa lisibilité.
- Ce projet est un travail académique et ne constitue pas un outil de diagnostic médical.

---

## 8. Structure du projet

```
.
├── FDAnalyse.ipynb      # Notebook complet (analyse et modèles)
├── heart excel.xlsx     # Jeu de données (à placer dans Google Drive)
└── README.md            # Ce document
```
