"""
=============================================================
08b — Clustering avec K-Means
=============================================================

CONCEPT CLÉ — K-MEANS :
    Algorithme non-supervisé qui regroupe des points en K clusters
    en minimisant la distance intra-cluster.

    ALGORITHME (itératif) :
        1. Initialiser K centroïdes aléatoirement
        2. Assigner chaque point au centroïde le plus proche
        3. Recalculer les centroïdes (moyenne des points du cluster)
        4. Répéter 2-3 jusqu'à convergence (les clusters ne changent plus)

    VISUALISATION :
        Avant :   ×  ○ ×  ○  × ○    (mélangé)
        Après :   [×××]  [○○○]       (groupé)

DIFFÉRENCE AVEC SVM :
    SVM      → supervisé (on connaît les labels à l'avance)
    K-Means  → non-supervisé (on découvre la structure des données)

PARAMÈTRE CLÉ : K (nombre de clusters)
    Trop petit K → clusters trop généraux
    Trop grand K → sur-segmentation
    → Méthode du "coude" (elbow method) pour choisir K optimal

AVEC WORD EMBEDDINGS :
    K-Means peut clustérer des textes représentés par leurs embeddings.
    → Les phrases sémantiquement proches se retrouvent dans le même cluster.
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────
# 1. K-MEANS SUR DES POINTS 2D
# ─────────────────────────────────────────────

def exercice1_kmeans_2d():
    """
    Exercice 1 : K-Means sur des points 2D.

    Visualisation simple pour comprendre comment K-Means
    regroupe des points dans un espace 2D.

    On peut voir visuellement que 3 groupes naturels existent :
        - Groupe bas-gauche  : [1,2], [1,3], [2,2]
        - Groupe haut-gauche : [1,8], [3,9]
        - Groupe haut-droite : [9,8], [8,8], [6,9], [4,7], [4,8]
    """
    print("=" * 45)
    print("EXERCICE 1 — K-Means sur points 2D")
    print("=" * 45)

    # Données : 10 points dans le plan
    points = [
        [1, 2], [1, 3], [2, 2],   # Cluster probable : bas-gauche
        [4, 7], [9, 8], [8, 8],   # Cluster probable : haut
        [1, 8], [3, 9], [6, 9], [4, 8]  # Mélangé → K-Means décide
    ]

    # K-Means avec 3 clusters
    # n_init=10 : essayer 10 initialisations différentes (prendre la meilleure)
    # random_state=0 : résultat reproductible
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    labels = kmeans.fit_predict(points)
    # labels = [0, 0, 0, 1, 2, 2, 1, 1, 2, 1] par exemple

    print(f"\nLabels attribués : {labels.tolist()}")
    print(f"Centroïdes :")
    for i, centroide in enumerate(kmeans.cluster_centers_):
        print(f"  Cluster {i} : ({centroide[0]:.2f}, {centroide[1]:.2f})")

    # ── Visualisation ──
    colors = ["red", "blue", "green"]
    plt.figure(figsize=(7, 5))

    for i, (x, y) in enumerate(points):
        couleur = colors[labels[i]]
        plt.scatter(x, y, color=couleur, s=80, zorder=3)
        plt.text(x + 0.1, y + 0.1, str([x, y]), fontsize=8, color=couleur)

    # Afficher les centroïdes
    for i, (cx, cy) in enumerate(kmeans.cluster_centers_):
        plt.scatter(cx, cy, color=colors[i], marker='X', s=200,
                    zorder=4, edgecolors='black', linewidth=1.5,
                    label=f"Centroïde {i}")

    plt.title("K-Means avec 3 clusters (×=centroïdes)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kmeans_2d.png", dpi=100)
    plt.show()


# ─────────────────────────────────────────────
# 2. K-MEANS SUR DES MOTS (FEATURES LINGUISTIQUES)
# ─────────────────────────────────────────────

def exercice2_kmeans_mots():
    """
    Exercice 2 : Clustering de mots avec des features simples.

    On représente chaque mot par :
        x = longueur du mot (nombre de caractères)
        y = nombre de chiffres dans le mot

    Ce clustering peut distinguer :
        - Mots ordinaires   : "chat", "Salut" → courts, sans chiffres
        - Noms propres      : "Sorbonne", "Paris" → longs, sans chiffres
        - Mots alpha-numériques : "python3", "2026" → avec chiffres
    """
    print("\n" + "=" * 45)
    print("EXERCICE 2 — K-Means sur mots (features linguistiques)")
    print("=" * 45)

    mots = ["chat", "Sorbonne", "b234", "voiture2026", "Lea", "2026",
            "75006Paris", "Maria89", "x", "Salut", "z1", "python3",
            "AB123456", "0611223344"]

    # Construire les features pour chaque mot
    points = []
    for mot in mots:
        x = len(mot)                              # Longueur totale
        y = sum(ch.isdigit() for ch in mot)       # Nombre de chiffres
        points.append([x, y])

    # Tester avec K=2, K=3, K=4
    for K in [2, 3]:
        print(f"\n  K = {K} clusters :")
        kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
        labels = kmeans.fit_predict(points)

        # Afficher les clusters
        clusters = {i: [] for i in range(K)}
        for mot, label in zip(mots, labels):
            clusters[label].append(mot)

        for k, membres in clusters.items():
            print(f"    Cluster {k} : {membres}")

    # ── Visualisation pour K=3 ──
    K = 3
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
    labels = kmeans.fit_predict(points)

    colors = plt.cm.tab10.colors  # Palette de 10 couleurs
    plt.figure(figsize=(8, 5))

    for i, (pt, mot) in enumerate(zip(points, mots)):
        plt.scatter(pt[0], pt[1], color=colors[labels[i]], s=50)
        plt.text(pt[0] + 0.1, pt[1] + 0.05, mot, fontsize=8)

    plt.xlabel("Longueur du mot")
    plt.ylabel("Nombre de chiffres")
    plt.title(f"K-Means sur mots (K={K})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kmeans_mots.png", dpi=100)
    plt.show()


# ─────────────────────────────────────────────
# 3. K-MEANS AVEC SENTENCE EMBEDDINGS
# ─────────────────────────────────────────────

def exercice3_kmeans_phrases():
    """
    Exercice 3 : Clustering de phrases avec embeddings sémantiques.

    On utilise SentenceTransformer pour représenter chaque phrase
    par un vecteur de 384 dimensions capturant son sens.

    PRINCIPE :
        "Le chat dort" et "Le chien sommeille" → vecteurs proches
        "La voiture roule" → vecteur plus éloigné des deux précédents

    Le modèle "paraphrase-multilingual-MiniLM-L12-v2" est multilingue
    → fonctionne pour le français, l'anglais, etc.

    NORMALISATION AVANT KMEANS :
        Diviser chaque vecteur par sa norme L2.
        → Tous les vecteurs ont une longueur de 1.
        → K-Means est équivalent à la similarité cosinus.
    """
    print("\n" + "=" * 45)
    print("EXERCICE 3 — K-Means avec Sentence Embeddings")
    print("=" * 45)

    # Installer si nécessaire :
    # pip install sentence-transformers

    try:
        from sentence_transformers import SentenceTransformer

        phrases = [
            # Thème 1 : Alimentation
            "Elle adore le chocolat chaud.",
            "J'achète un pain au chocolat.",
            "Je mange un sandwich au jambon.",
            "La baguette est croustillante.",
            "Le fromage vient de la ferme.",
            # Thème 2 : Animaux
            "Le chien court dans le parc.",
            "Le lion vit en Afrique.",
            "L'oiseau vole dans le ciel.",
            "Le poisson nage dans l'eau.",
            # Thème 3 : Université
            "La Sorbonne est une université célèbre.",
            "La bibliothèque est grande.",
            "Les étudiants étudient à la Sorbonne.",
            # Thème 4 : IA
            "Les algorithmes prennent des décisions.",
            "L'IA analyse des images.",
            "L'IA apprend à partir des données.",
            "Python est utilisé en intelligence artificielle.",
        ]

        K = 4  # On sait qu'il y a 4 thèmes

        print("Chargement du modèle SentenceTransformer...")
        modele_embeddings = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Encoder toutes les phrases en vecteurs
        print("Encodage des phrases...")
        X = modele_embeddings.encode(phrases)
        print(f"Dimension des embeddings : {X.shape}")
        # → (16, 384) : 16 phrases, 384 dimensions

        # Normalisation L2 : vecteurs de norme = 1
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # K-Means
        kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)

        # Afficher les clusters
        print(f"\nRésultats K-Means (K={K}) :")
        clusters = {i: [] for i in range(K)}
        for phrase, label in zip(phrases, labels):
            clusters[label].append(phrase)

        for k, membres in clusters.items():
            print(f"\n  Cluster {k} :")
            for phrase in membres:
                print(f"    - {phrase}")

    except ImportError:
        print("  sentence-transformers non installé.")
        print("  → pip install sentence-transformers")


# ─────────────────────────────────────────────
# 4. VISUALISATION PCA + K-MEANS
# ─────────────────────────────────────────────

def exercice4_pca_visualisation():
    """
    Exercice 4 : Réduire à 2D avec PCA pour visualiser les clusters.

    PROBLÈME : Les embeddings ont 384 dimensions → impossible à visualiser.
    SOLUTION : PCA (Analyse en Composantes Principales)
        → Projeter dans un espace 2D en conservant le maximum de variance.

    Ce n'est qu'une approximation : certaines séparations peuvent
    être invisibles en 2D mais claires en 384D.
    """
    print("\n" + "=" * 45)
    print("EXERCICE 4 — K-Means + PCA (visualisation 2D)")
    print("=" * 45)

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA

        phrases = [
            "L'hôpital accueille les malades.",
            "L'infirmière soigne les malades.",
            "Le médecin travaille à l'hôpital.",
            "La banane est un fruit jaune.",
            "La pomme est un fruit sucré.",
            "Le raisin est un petit fruit.",
            "La voiture est un moyen de transport.",
            "Le bus est un transport public.",
            "Le train est un transport rapide.",
            "Le vélo est un moyen de transport écologique.",
        ]

        print("Chargement et encodage...")
        modele = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        X = modele.encode(phrases)

        # Normalisation
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # K-Means (3 clusters : médical, fruits, transport)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)

        # PCA : réduire de 384D à 2D pour la visualisation
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        variance_expliquee = pca.explained_variance_ratio_.sum() * 100

        print(f"Variance expliquée par les 2 premières composantes : {variance_expliquee:.1f}%")

        # Visualisation
        colors = ["red", "blue", "green"]
        plt.figure(figsize=(9, 6))

        for i, (phrase, label) in enumerate(zip(phrases, labels)):
            plt.scatter(X_2d[i, 0], X_2d[i, 1],
                       color=colors[label], s=80, zorder=3)
            # Texte court pour la lisibilité
            texte_court = phrase[:30] + "..." if len(phrase) > 30 else phrase
            plt.annotate(texte_court, (X_2d[i, 0], X_2d[i, 1]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

        plt.title(f"K-Means + PCA ({variance_expliquee:.0f}% de variance)")
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("kmeans_pca.png", dpi=100)
        plt.show()
        print("Graphique sauvegardé → kmeans_pca.png")

    except ImportError:
        print("  sentence-transformers non installé.")
        print("  → pip install sentence-transformers")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    exercice1_kmeans_2d()
    exercice2_kmeans_mots()
    exercice3_kmeans_phrases()
    exercice4_pca_visualisation()
