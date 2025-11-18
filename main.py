import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from train_model import train_model
from preprocess_data import preprocess_data

# Charger le dataset
iris = pd.read_csv(r"C:\Users\lawan\TP_Git\InputData\Iris.csv")

# Taille du test
test_size = 0.4

# Prétraitement
train, test = preprocess_data(iris, test_size)

# Données d'entraînement et de test
train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
train_y = train.Species
test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
test_y = test.Species

# Modèle SVM
model = svm.SVC(kernel='linear', random_state=42)

# Entraînement via ta fonction
prediction = train_model(train_X, train_y, test_X, model)

# -------------------------------
# Calcul de la précision (accuracy)
# -------------------------------
acc = accuracy_score(test_y, prediction)
print(f" Accuracy du modèle sur le jeu de test : {acc:.4f}")

# -------------------------------
# Courbe d'apprentissage
# -------------------------------
train_sizes, train_scores, test_scores = learning_curve(
    model,
    train_X,
    train_y,
    cv=5,                  # validation croisée à 5 plis
    scoring='accuracy',
    n_jobs=-1,
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
)

# Moyennes et écarts-types
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# Tracer la courbe
plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, label='Score entraînement', color='blue')
plt.plot(train_sizes, test_mean, label='Score validation (CV)', color='green')

# Zones d’écart-type
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.2)

# Annotation de la précision finale
plt.axhline(y=acc, color='red', linestyle='--', label=f'Accuracy test = {acc:.2f}')

plt.title("Courbe d'apprentissage - SVM sur Iris")
plt.xlabel("Taille de l'échantillon d'entraînement")
plt.ylabel("Précision (Accuracy)")
plt.legend(loc="best")
plt.grid(True)
plt.show()
print("hello lawa")