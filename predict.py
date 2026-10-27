import joblib
import pandas as pd

# Chargement du modèle et de la liste des colonnes utilisées à l'entraînement
modele = joblib.load("model.pkl")
colonnes = joblib.load("colonnes.pkl")


def predire_prix(logement):
    df = pd.DataFrame([logement])
    df = pd.get_dummies(df, columns=["neighbourhood_group", "room_type"])
    df = df.reindex(columns=colonnes, fill_value=0)
    return modele.predict(df)[0]


# Exemple : un logement entier à Manhattan
nouveau_logement = {
    "neighbourhood_group": "Manhattan",
    "room_type": "Entire home/apt",
    "latitude": 40.75,
    "longitude": -73.98,
    "minimum_nights": 2,
    "number_of_reviews": 20,
    "reviews_per_month": 1.5,
    "calculated_host_listings_count": 1,
    "availability_365": 200
}

prix = predire_prix(nouveau_logement)
print(f"Prix prédit : {prix:.0f} $ par nuit")