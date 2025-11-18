from sklearn.model_selection import train_test_split

def preprocess_data(data, testSize):
    """
    Nettoie, met en forme les données et prépare les ensembles
    d'entraînement et de test.
    Supprime la colonne 'Id' des données.
    """
    # Supprimer la colonne 'Id' si elle existe
    if 'Id' in data.columns:
        data = data.drop(columns=['Id'])

    # Séparer les données en train et test
    train, test = train_test_split(data, test_size=testSize, random_state=42, shuffle=True)

    return train, test
 