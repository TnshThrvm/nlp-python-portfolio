"""
=============================================================
08c — Réseaux de neurones avec sklearn (MLP)
=============================================================

CONCEPT CLÉ — RÉSEAU DE NEURONES MULTICOUCHE (MLP) :
    Un réseau de neurones mimique le fonctionnement du cerveau :
    des neurones artificiels organisés en couches qui apprennent
    à reconnaître des patterns dans les données.

    ARCHITECTURE :
        Couche d'entrée    → données brutes (ex: pixels d'image)
            ↓
        Couche(s) cachée(s) → features abstraites
            ↓
        Couche de sortie   → probabilités pour chaque classe

    COMMENT ÇA APPREND :
        1. Forward pass  : calculer la prédiction
        2. Calcul de la perte (cross-entropie)
        3. Backward pass : ajuster les poids par gradient descent
        4. Répéter jusqu'à convergence

    DATASET UTILISÉ : MNIST chiffres manuscrits
        - 1797 images de chiffres 0-9
        - Chaque image = 8×8 pixels = 64 features
        - 10 classes (un par chiffre)

GÉNÉRALISATION :
    Le même MLP peut classifier des images, du texte, de l'audio...
    Il suffit de changer les features en entrée.

HYPERPARAMÈTRES CLÉS :
    hidden_layer_sizes : architecture des couches cachées
        (10,)      → 1 couche de 10 neurones
        (100, 50)  → 2 couches (100 puis 50 neurones)
    max_iter : nombre maximum d'epochs d'entraînement
=============================================================
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
# 1. CHARGEMENT ET EXPLORATION DES DONNÉES
# ─────────────────────────────────────────────

def charger_donnees():
    """
    Charge le dataset MNIST réduit (chiffres 0-9, images 8×8).

    Ce dataset est intégré à scikit-learn → pas besoin de télécharger.

    Retour :
        tuple : (data, X, y) où
            - data : objet sklearn avec .images, .target, .DESCR
            - X    : features aplaties, shape (1797, 64)
            - y    : labels, shape (1797,)

    Structure de data :
        data.images   → (1797, 8, 8) : les images originales
        data.target   → (1797,)      : les chiffres (0 à 9)
        data.data     → (1797, 64)   : images aplaties
    """
    # Chargement du dataset intégré à sklearn
    data = datasets.load_digits()

    print("=== Dataset MNIST réduit ===")
    print(f"Nombre d'images   : {len(data.images)}")
    print(f"Forme d'une image : {data.images[0].shape}")
    print(f"Classes           : {data.target_names.tolist()}")

    # Aplatir les images 8×8 en vecteurs de 64 features
    # reshape(-1) = aplatir en 1D, len() = nombre d'images
    X = data.images.reshape((len(data.images), -1))
    y = data.target

    print(f"\nAprès aplatissement :")
    print(f"  X.shape = {X.shape}   (1797 images × 64 pixels)")
    print(f"  y.shape = {y.shape}   (1797 labels)")

    return data, X, y


def afficher_exemples(data, n: int = 10):
    """
    Affiche les premières images du dataset avec leur label.

    Paramètres :
        data : dataset sklearn
        n (int) : nombre d'images à afficher
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.ravel()

    for i in range(min(n, 10)):
        axes[i].imshow(data.images[i], cmap='gray')
        axes[i].set_title(f"Label: {data.target[i]}", fontsize=10)
        axes[i].axis('off')

    plt.suptitle("Exemples d'images du dataset MNIST réduit", y=1.02)
    plt.tight_layout()
    plt.savefig("mnist_exemples.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("Graphique sauvegardé → mnist_exemples.png")


# ─────────────────────────────────────────────
# 2. PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────

def preparer_donnees(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Sépare les données en jeux d'entraînement et de test.

    RÈGLE GÉNÉRALE :
        - 80% pour l'entraînement (le modèle apprend dessus)
        - 20% pour le test (évaluation sur données jamais vues)

    Pourquoi ne pas tout utiliser pour l'entraînement ?
        → Le modèle pourrait mémoriser les données (overfitting)
        → Il ne généraliserait pas sur de nouvelles données

    Paramètres :
        X (np.ndarray)      : features
        y (np.ndarray)      : labels
        test_size (float)   : proportion pour le test (0 à 1)
        random_state (int)  : graine aléatoire pour la reproductibilité

    Retour :
        tuple : (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print("=== Séparation train/test ===")
    print(f"  Entraînement : {X_train.shape[0]} images ({(1-test_size)*100:.0f}%)")
    print(f"  Test         : {X_test.shape[0]} images ({test_size*100:.0f}%)")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 3. ENTRAÎNEMENT DU RÉSEAU DE NEURONES
# ─────────────────────────────────────────────

def creer_et_entrainer_mlp(X_train, y_train,
                             hidden_layers=(10,),
                             max_iter: int = 300):
    """
    Crée et entraîne un réseau de neurones MLP (Multi-Layer Perceptron).

    Paramètres :
        X_train (np.ndarray)    : données d'entraînement
        y_train (np.ndarray)    : labels d'entraînement
        hidden_layers (tuple)   : architecture des couches cachées
                                  (10,) = 1 couche de 10 neurones
                                  (100, 50) = 2 couches
        max_iter (int)          : nombre max d'itérations d'entraînement

    Retour :
        MLPClassifier : modèle entraîné

    Architecture avec (10,) :
        64 features → 10 neurones cachés → 10 classes (chiffres 0-9)

    Architecture avec (100,) :
        64 features → 100 neurones cachés → 10 classes
        (plus de capacité, potentiellement plus précis mais plus lent)
    """
    print(f"\n=== Création du MLP : couches cachées = {hidden_layers} ===")

    # Création du classifieur MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        random_state=42   # Reproductibilité
    )

    print("Entraînement en cours...")
    mlp.fit(X_train, y_train)
    print(f"Entraînement terminé en {mlp.n_iter_} itérations")

    return mlp


# ─────────────────────────────────────────────
# 4. ÉVALUATION
# ─────────────────────────────────────────────

def evaluer_modele(mlp, X_test, y_test):
    """
    Évalue le modèle entraîné sur les données de test.

    Métriques calculées :
        - Accuracy (précision globale) : % de bonnes prédictions
        - Rapport de classification : précision/rappel/F1 par chiffre

    Paramètres :
        mlp         : modèle MLPClassifier entraîné
        X_test      : features de test
        y_test      : vrais labels de test

    Retour :
        tuple : (predictions, accuracy)
    """
    predictions = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n=== Résultats d'évaluation ===")
    print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Rapport détaillé par chiffre
    print("\nRapport de classification :")
    print(classification_report(y_test, predictions,
                                  target_names=[str(i) for i in range(10)]))

    return predictions, accuracy


def comparer_predictions(predictions, y_test, data, X_test, n: int = 10):
    """
    Affiche une comparaison visuelle entre les prédictions et les vraies valeurs.

    Met en évidence les erreurs de classification.

    Paramètres :
        predictions  : prédictions du modèle
        y_test       : vraies valeurs
        data         : dataset original (pour afficher les images)
        X_test       : features de test (pour reconstruire les images 8×8)
        n (int)      : nombre d'exemples à afficher
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(min(n, len(y_test))):
        # Reconstituer l'image 8×8 depuis le vecteur 64D
        image = X_test[i].reshape(8, 8)
        axes[i].imshow(image, cmap='gray')

        # Couleur selon si correct (vert) ou erreur (rouge)
        est_correct = (predictions[i] == y_test[i])
        couleur = 'green' if est_correct else 'red'
        symbole = '✓' if est_correct else '✗'

        axes[i].set_title(
            f"Prédit: {predictions[i]} {symbole}\nVrai: {y_test[i]}",
            color=couleur, fontsize=9
        )
        axes[i].axis('off')

    plt.suptitle("Prédictions (vert=correct, rouge=erreur)", y=1.02)
    plt.tight_layout()
    plt.savefig("mlp_predictions.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("Graphique sauvegardé → mlp_predictions.png")


# ─────────────────────────────────────────────
# 5. COMPARAISON DE DIFFÉRENTES ARCHITECTURES
# ─────────────────────────────────────────────

def comparer_architectures(X_train, X_test, y_train, y_test):
    """
    Compare plusieurs architectures de MLP pour voir l'effet
    du nombre de neurones et de couches sur la performance.

    Architectures testées :
        (5,)      → 1 couche, très petit
        (10,)     → 1 couche, petit (configuration de base)
        (50,)     → 1 couche, moyen
        (100,)    → 1 couche, grand
        (100, 50) → 2 couches
    """
    print("\n=== Comparaison d'architectures ===")
    architectures = [(5,), (10,), (50,), (100,), (100, 50)]

    resultats = []
    for arch in architectures:
        mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=300,
                            random_state=42)
        mlp.fit(X_train, y_train)
        preds = mlp.predict(X_test)
        acc = accuracy_score(y_test, preds)
        resultats.append((str(arch), acc))
        print(f"  Architecture {str(arch):12} → Accuracy : {acc*100:.1f}%")

    # Visualisation
    noms, scores = zip(*resultats)
    plt.figure(figsize=(8, 5))
    barres = plt.bar(noms, [s * 100 for s in scores],
                     color='steelblue', edgecolor='black')

    # Afficher les valeurs au-dessus des barres
    for barre, score in zip(barres, scores):
        plt.text(barre.get_x() + barre.get_width() / 2,
                 barre.get_height() + 0.5,
                 f"{score*100:.1f}%", ha='center', fontsize=9)

    plt.xlabel("Architecture (couches cachées)")
    plt.ylabel("Accuracy (%)")
    plt.title("Comparaison des architectures MLP sur MNIST")
    plt.ylim(0, 105)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("mlp_comparaison.png", dpi=100)
    plt.show()
    print("Graphique sauvegardé → mlp_comparaison.png")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE — PIPELINE COMPLET
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("PIPELINE COMPLET : MLP sur MNIST\n")

    # Étape 1 : Charger les données
    data, X, y = charger_donnees()

    # Étape 2 : Afficher quelques exemples
    afficher_exemples(data)

    # Étape 3 : Séparer train/test
    X_train, X_test, y_train, y_test = preparer_donnees(X, y)

    # Étape 4 : Entraîner le MLP de base
    mlp = creer_et_entrainer_mlp(X_train, y_train,
                                   hidden_layers=(10,), max_iter=300)

    # Étape 5 : Évaluer
    predictions, accuracy = evaluer_modele(mlp, X_test, y_test)

    # Étape 6 : Visualiser les prédictions
    comparer_predictions(predictions, y_test, data, X_test)

    # Étape 7 : Comparer différentes architectures
    comparer_architectures(X_train, X_test, y_train, y_test)

    print("\n=== RÉSUMÉ ===")
    print(f"Meilleure accuracy obtenue avec (10,) : {accuracy*100:.1f}%")
    print("""
WORKFLOW GÉNÉRIQUE MLP :
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    # 1. Charger
    data = datasets.load_digits()
    X = data.images.reshape(len(data.images), -1)
    y = data.target

    # 2. Séparer
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 3. Entraîner
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    mlp.fit(X_train, y_train)

    # 4. Évaluer
    predictions = mlp.predict(X_test)
    print(accuracy_score(y_test, predictions))
    """)
