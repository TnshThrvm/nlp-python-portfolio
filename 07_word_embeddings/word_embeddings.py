"""
=============================================================
07 — Word Embeddings : Représentations vectorielles des mots
=============================================================

CONCEPT CLÉ — WORD EMBEDDINGS :
    Représenter chaque mot par un VECTEUR de nombres réels
    dans un espace à N dimensions (typiquement 100 à 300 dims).

    INTUITION FONDAMENTALE :
        "Un mot est défini par les mots qui l'entourent."
        → chat et chien apparaissent dans des contextes similaires
        → leurs vecteurs sont proches dans l'espace vectoriel

    OPÉRATIONS VECTORIELLES FASCINANTES :
        king - man + woman ≈ queen
        Paris - France + Italy ≈ Rome
        → Les embeddings capturent des analogies sémantiques !

    MÉTHODES PRINCIPALES :
        Word2Vec (Google, 2013) → entraîné sur le contexte local
        GloVe    (Stanford, 2014) → co-occurrences globales
        FastText (Facebook, 2017) → sous-mots (gère les mots inconnus)

    SIMILARITÉ COSINUS :
        Mesure l'angle entre deux vecteurs.
        sim(u, v) = (u · v) / (|u| × |v|)
        → 1.0 = identiques, 0.0 = orthogonaux, -1.0 = opposés

GÉNÉRALISATION :
    Les mêmes opérations s'appliquent aux phrase embeddings,
    aux embeddings de documents, et même aux images dans les
    modèles multimodaux (CLIP, DALL-E).
=============================================================
"""

import numpy as np


# ─────────────────────────────────────────────
# 1. OPÉRATIONS VECTORIELLES FONDAMENTALES
# ─────────────────────────────────────────────

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs.

    sim(u, v) = (u · v) / (||u|| × ||v||)

    Paramètres :
        u, v (np.ndarray) : vecteurs de même dimension

    Retour :
        float : similarité entre -1 et 1
            1.0  → mots très proches sémantiquement (ex: chat/chien)
            0.0  → mots non liés (ex: chat/voiture)
           -1.0  → mots opposés (ex: bon/mauvais dans certains espaces)

    Exemple :
        >>> u = np.array([1, 0, 0])
        >>> v = np.array([0, 1, 0])
        >>> cosine_similarity(u, v)
        0.0   # Vecteurs orthogonaux → pas de similarité

        >>> cosine_similarity(u, u)
        1.0   # Un vecteur est identique à lui-même
    """
    # Produit scalaire : u · v = Σ(ui × vi)
    produit_scalaire = np.dot(u, v)

    # Normes L2 : ||u|| = sqrt(Σ ui²)
    norme_u = np.linalg.norm(u)
    norme_v = np.linalg.norm(v)

    # Éviter la division par zéro (vecteur nul)
    if norme_u == 0 or norme_v == 0:
        return 0.0

    return produit_scalaire / (norme_u * norme_v)


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalise un vecteur pour qu'il ait une norme L2 = 1.

    Après normalisation, le vecteur pointe dans la même direction
    mais avec une longueur unitaire. Utile pour comparer des
    directions sans être influencé par la magnitude.

    Paramètre :
        v (np.ndarray) : vecteur à normaliser

    Retour :
        np.ndarray : vecteur normalisé (même direction, norme = 1)

    Exemple :
        >>> normalize(np.array([3, 4]))
        array([0.6, 0.8])   # sqrt(0.36 + 0.64) = 1.0
    """
    norme = np.linalg.norm(v)
    if norme == 0:
        return v  # Vecteur nul → on ne peut pas normaliser
    return v / norme


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calcule la distance euclidienne entre deux vecteurs.

    dist(u, v) = ||u - v|| = sqrt(Σ (ui - vi)²)

    Différence avec la similarité cosinus :
        - Distance euclidienne → tient compte de la magnitude
        - Similarité cosinus   → seulement la direction

    Paramètres :
        u, v (np.ndarray) : vecteurs de même dimension

    Retour :
        float : distance (positif, 0 si identiques)

    Exemple :
        >>> euclidean_distance(np.array([0,0]), np.array([3,4]))
        5.0   # Théorème de Pythagore : sqrt(9+16) = 5
    """
    return np.linalg.norm(u - v)


# ─────────────────────────────────────────────
# 2. CHARGEMENT DE MODÈLES PRÉ-ENTRAÎNÉS
# ─────────────────────────────────────────────

def charger_word2vec(chemin_modele: str):
    """
    Charge un modèle Word2Vec ou GloVe avec gensim.

    Paramètre :
        chemin_modele (str) : chemin vers le fichier du modèle
            - Binaire Word2Vec (.bin) : binary=True
            - Texte GloVe (.txt)     : binary=False, no_header=True

    Retour :
        KeyedVectors : modèle chargé (similaire à un dictionnaire {mot: vecteur})

    Exemple d'utilisation après chargement :
        model['cat']                    → vecteur 300D de "cat"
        model.most_similar('cat')       → 10 mots les plus proches
        model.similarity('cat', 'dog')  → similarité cosinus
        model.doesnt_match(['cat', 'dog', 'car']) → "car" (l'intrus)
    """
    from gensim.models import KeyedVectors

    # Détecter le format depuis l'extension
    if chemin_modele.endswith('.bin'):
        # Format Word2Vec binaire (Google News vectors)
        modele = KeyedVectors.load_word2vec_format(
            chemin_modele,
            binary=True
        )
    else:
        # Format GloVe texte (avec no_header car pas d'en-tête)
        modele = KeyedVectors.load_word2vec_format(
            chemin_modele,
            binary=False,
            no_header=True
        )

    return modele


# ─────────────────────────────────────────────
# 3. ANALOGIES ET OPÉRATIONS
# ─────────────────────────────────────────────

def tester_analogie(modele, positifs: list, negatifs: list, topk: int = 5) -> list:
    """
    Teste une analogie vectorielle du type : A - B + C = ?

    Formule : king - man + woman = queen
        positifs = ['king', 'woman']  (mots à additionner)
        negatifs = ['man']            (mots à soustraire)

    Paramètres :
        modele    : modèle gensim chargé
        positifs  : liste de mots à additionner
        negatifs  : liste de mots à soustraire
        topk (int): nombre de résultats à retourner

    Retour :
        list : [(mot, score), ...] les topk mots les plus proches

    Exemples d'analogies classiques :
        king - man + woman = queen
        Paris - France + Italy = Rome
        walking - walk + swim = swimming
    """
    resultats = modele.most_similar(
        positive=positifs,
        negative=negatifs,
        topn=topk
    )
    return resultats


def visualiser_embeddings_tsne(modele, mots: list) -> None:
    """
    Visualise des embeddings en 2D avec t-SNE.

    t-SNE (t-distributed Stochastic Neighbor Embedding) est une
    technique de réduction de dimensionnalité qui préserve les
    distances locales : les mots similaires restent proches.

    Paramètres :
        modele : modèle gensim
        mots (list) : liste de mots à visualiser

    Exemple de mots typiques pour visualiser les clusters :
        mots = ['cat', 'dog', 'kitten', 'puppy',
                'car', 'truck', 'vehicle',
                'king', 'queen', 'prince']
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Récupérer les vecteurs pour les mots demandés
    vecteurs = np.array([modele[mot] for mot in mots if mot in modele])
    mots_trouves = [mot for mot in mots if mot in modele]

    # Réduire à 2 dimensions avec t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(mots)-1))
    coords = tsne.fit_transform(vecteurs)

    # Affichage
    plt.figure(figsize=(10, 8))
    for i, mot in enumerate(mots_trouves):
        plt.scatter(coords[i, 0], coords[i, 1], s=50)
        plt.annotate(mot, (coords[i, 0], coords[i, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=10)

    plt.title("Visualisation t-SNE des word embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("tsne_embeddings.png", dpi=100)
    plt.show()
    print("Graphique sauvegardé → tsne_embeddings.png")


# ─────────────────────────────────────────────
# 4. EXEMPLE SANS MODÈLE PRÉ-ENTRAÎNÉ
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Démonstration des opérations vectorielles ──

    print("=== Opérations vectorielles sur des vecteurs simulés ===\n")

    # Vecteurs simulés (en réalité, ce seraient des vecteurs Word2Vec de 300 dims)
    # On simule des embeddings 4D pour illustrer
    vecteurs_simules = {
        "chat":    np.array([0.8,  0.1, -0.2,  0.5]),
        "chien":   np.array([0.7,  0.2, -0.1,  0.6]),
        "voiture": np.array([-0.1, 0.9,  0.8, -0.3]),
        "moto":    np.array([-0.2, 0.8,  0.7, -0.2]),
        "maison":  np.array([0.1, -0.3,  0.2,  0.9]),
    }

    # Test de similarité cosinus
    print("Similarités cosinus :")
    paires = [
        ("chat", "chien"),     # Deux animaux → devrait être élevé
        ("chat", "voiture"),   # Animal vs véhicule → devrait être bas
        ("voiture", "moto"),   # Deux véhicules → devrait être élevé
    ]
    for m1, m2 in paires:
        sim = cosine_similarity(vecteurs_simules[m1], vecteurs_simules[m2])
        print(f"  sim({m1}, {m2}) = {sim:.3f}")

    # Test de distance euclidienne
    print("\nDistances euclidiennes :")
    for m1, m2 in paires:
        dist = euclidean_distance(vecteurs_simules[m1], vecteurs_simules[m2])
        print(f"  dist({m1}, {m2}) = {dist:.3f}")

    # Test de normalisation
    print("\nNormalisation :")
    v = np.array([3, 4])
    v_norm = normalize(v)
    print(f"  Vecteur original  : {v}")
    print(f"  Vecteur normalisé : {v_norm}")
    print(f"  Norme après normalisation : {np.linalg.norm(v_norm):.6f}")  # Doit être 1.0

    # ── Utilisation avec gensim ──
    print("\n=== Code pour utiliser avec un vrai modèle gensim ===")
    print("""
    from gensim.models import KeyedVectors

    # Charger le modèle
    model = KeyedVectors.load_word2vec_format('glove.6B.100d.txt',
                                               binary=False, no_header=True)

    # Vecteur d'un mot
    vecteur_cat = model['cat']
    print("Dimension :", len(vecteur_cat))  # → 100

    # 10 mots les plus similaires
    for mot, score in model.most_similar('cat', topn=10):
        print(f"  {mot}: {score:.3f}")

    # Analogie king - man + woman = ?
    resultats = model.most_similar(positive=['king', 'woman'],
                                    negative=['man'], topn=5)
    print(resultats)  # → [('queen', 0.85), ...]

    # Trouver l'intrus
    intrus = model.doesnt_match(['cat', 'dog', 'bird', 'car'])
    print("Intrus :", intrus)  # → 'car'
    """)
