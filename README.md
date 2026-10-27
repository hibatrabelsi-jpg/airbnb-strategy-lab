# Airbnb Strategy Lab

> Analyser un marché, prédire le juste prix, décider avec des chiffres.

## Contexte

Un fonds d'investissement immobilier souhaite se positionner sur la location courte durée (Airbnb) à New York. Le directeur d'investissement veut un outil capable de **repérer les biens sous-évalués** et de le **prouver avec des chiffres**.

Ce projet passe d'une donnée brute et imparfaite à un outil d'aide à la décision qui :
1. **décrit le marché** (analyse exploratoire) ;
2. **prédit le juste prix** d'une nuit (Machine Learning) ;
3. **repère les opportunités** (détecteur de bonnes affaires).

## Données

- **Source :** [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (AB_NYC_2019, Kaggle)
- **Taille :** 48 895 annonces, 16 colonnes (prix, arrondissement, quartier, type de logement, avis, disponibilité, coordonnées GPS…)

## Structure du projet

```
airbnb-strategy-lab/
├── data/
│   ├── AB_NYC_2019.csv            # données brutes
│   └── AB_NYC_2019_clean.csv      # données nettoyées (sortie du notebook 1)
├── notebooks/
│   ├── 01_analyse_exploratoire.ipynb
│   └── 02_modelisation.ipynb
├── presentation/                  # support de présentation
├── model.pkl                      # modèle entraîné (Random Forest)
├── colonnes.pkl                   # colonnes attendues par le modèle
├── predict.py                     # script de prédiction
├── requirements.txt
└── README.md
```

## Démarche

### 1. Analyse exploratoire (`01_analyse_exploratoire.ipynb`)

**Nettoyage**
- `reviews_per_month` : valeurs manquantes remplacées par 0 (logements sans avis).
- `name` / `host_name` : valeurs manquantes remplacées par « Inconnu ».
- `last_review` : convertie en date (les valeurs manquantes sont conservées).
- Suppression des valeurs aberrantes : prix à 0 \$, prix au-delà du 99ᵉ percentile (799 \$), `minimum_nights` > 30 (au-delà, ce n'est plus de la location courte durée).
- **Résultat : 47 692 annonces conservées (2,46 % supprimées).**

**Enrichissement**
- `revenu_estime` = `price` × `reviews_per_month` : un proxy du revenu mensuel.
- `categorie_prix` : Low-cost (≤ 80 \$) / Standard (80–150 \$) / Luxe (> 150 \$), en 3 groupes de taille égale.

**Principaux enseignements**
- Les prix les plus élevés se concentrent à Manhattan.
- **Le quartier le plus cher n'est pas le plus rentable** : Manhattan a le prix médian le plus haut (149 \$) mais l'un des revenus estimés les plus faibles. Le Queens, le Bronx et Staten Island sont moins chers mais plus souvent loués.
- Un logement entier coûte en médiane 160 \$, contre 70 \$ pour une chambre privée.

### 2. Modélisation (`02_modelisation.ipynb`)

- **Target :** `price`
- **Features :** arrondissement, type de logement, latitude, longitude, nuits minimum, nombre d'avis, avis par mois, nombre d'annonces de l'hôte, disponibilité.
- **Encodage :** `get_dummies` pour les variables catégorielles.
- **Découpage :** 80 % entraînement / 20 % test (`random_state=42`).

**⚠️ Fuite de données évitée :** `revenu_estime` et `categorie_prix` sont toutes deux calculées à partir du prix. Les utiliser reviendrait à donner la réponse au modèle : elles ont été exclues des features.

## Résultats

| Modèle | MAE | RMSE | R² entraînement | R² test |
|---|---|---|---|---|
| Régression linéaire (baseline) | 52,76 \$ | 83,91 \$ | 0,367 | 0,363 |
| Random Forest | 46,43 \$ | 76,54 \$ | 0,929 | 0,470 |
| **Random Forest optimisé** | **45,20 \$** | **75,30 \$** | **0,689** | **0,487** |

- Le Random Forest par défaut est en **sur-apprentissage** (R² de 0,93 à l'entraînement contre 0,47 en test).
- En limitant la profondeur des arbres (`max_depth=15`, `min_samples_leaf=5`), le modèle optimisé réduit fortement cet écart et obtient le meilleur score en test : **c'est le modèle retenu.**
- **Variables les plus importantes :** le type de logement (38 %) et la localisation GPS (33 %).

## Détecteur de bonnes affaires

En comparant le prix affiché au prix prédit sur le jeu de test (logements avec au moins 10 avis), le **Top 10 des biens sous-évalués** est composé uniquement de **logements entiers**, dont 8 à Manhattan et 2 à Brooklyn. Ils sont affichés en moyenne à **115 \$** la nuit pour un juste prix estimé à **294 \$**, soit un écart d'environ **+180 \$ par nuit**.

## Recommandation

1. **Opportunités immédiates :** les biens du Top 10 offrent un potentiel d'environ +180 \$ par nuit, soit environ 18 000 \$ de revenu supplémentaire par an et par bien sur une base de 100 nuits louées.
2. **Stratégie de fond :** cibler les **logements entiers dans le Queens**, qui offrent le meilleur revenu estimé sur un marché fiable (5 598 annonces).
3. **Prudence :** le modèle se trompe en moyenne de 45 \$ et ignore la surface, l'état ou les équipements. Chaque opportunité doit être vérifiée par une visite.

## Utilisation

```bash
# Installer les dépendances
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Prédire le prix d'un logement
python predict.py
```

Exemple de sortie :

```
Prix prédit : 282 $ par nuit
```

Pour tester un autre logement, modifier le dictionnaire `nouveau_logement` dans `predict.py`.

## Technologies

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · joblib · Jupyter
