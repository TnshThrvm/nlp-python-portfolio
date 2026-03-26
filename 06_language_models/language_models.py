"""
=============================================================
06 — Modèles de langue : Unigramme et Bigramme
=============================================================

CONCEPT CLÉ — MODÈLE DE LANGUE :
    Un modèle de langue assigne une probabilité à une séquence de mots.
    Il répond à la question : "Quelle est la probabilité que ce texte
    soit naturel dans cette langue ?"

    MODÈLE UNIGRAMME (indépendance totale) :
        P(w1, w2, ..., wn) = P(w1) × P(w2) × ... × P(wn)
        Chaque mot est considéré INDÉPENDANT des autres.
        P(w) = nb_occurrences(w) / total_mots

    MODÈLE BIGRAMME (contexte du mot précédent) :
        P(w1, w2, ..., wn) = P(w1) × P(w2|w1) × P(w3|w2) × ...
        Chaque mot dépend du mot PRÉCÉDENT.
        P(wi | wi-1) = nb_occurrences(wi-1, wi) / nb_occurrences(wi-1)

    PERPLEXITÉ :
        Mesure à quel point un modèle est "surpris" par un texte.
        Basse perplexité = modèle bien adapté au texte.
        PPL = exp( -1/N × Σ log P(wi | contexte) )

GÉNÉRALISATION :
    Unigramme → rapide, aveugle au contexte
    Bigramme  → tient compte du contexte local
    N-gramme  → généralisation à N-1 mots de contexte
    Réseau de neurones (LLM) → contexte de longueur arbitraire

SMOOTHING (lissage) :
    Problème : si un mot n'a jamais été vu à l'entraînement,
    P(mot) = 0, ce qui annule toute la probabilité de la séquence.
    Solution : attribuer une probabilité minimale aux mots inconnus.
=============================================================
"""

import math
import numpy as np
from collections import Counter


# ─────────────────────────────────────────────
# 0. FONCTIONS NUMÉRIQUES DE BASE
# ─────────────────────────────────────────────

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Fonction softmax : transforme un vecteur de scores en
    distribution de probabilités (somme = 1).

    Formule : softmax(z_i) = exp(z_i) / Σ exp(z_j)

    Astuce numérique : on soustrait max(z) avant exp() pour éviter
    les overflow (exp de grands nombres = infini).

    Paramètre :
        z (np.ndarray) : vecteur de scores réels (logits)

    Retour :
        np.ndarray : vecteur de probabilités, même taille que z

    Exemple :
        >>> softmax(np.array([2.0, 1.0, 0.1]))
        array([0.659, 0.242, 0.099])   # Somme ≈ 1.0
    """
    # Soustraction du maximum pour la stabilité numérique
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule la perte cross-entropie entre une vraie distribution
    et une distribution prédite.

    Formule : CE = -Σ y_true_i × log(y_pred_i)

    Paramètres :
        y_true (np.ndarray) : vraie distribution (one-hot ou probabilités)
        y_pred (np.ndarray) : distribution prédite

    Retour :
        float : valeur de la perte (plus c'est petit, mieux c'est)

    Exemple :
        y_true = np.array([0, 1, 0])   # La vraie classe est 1
        y_pred = np.array([0.1, 0.7, 0.2])
        >>> cross_entropy(y_true, y_pred)
        0.357   # -log(0.7)
    """
    # Clip pour éviter log(0) = -infini
    y_pred_safe = np.clip(y_pred, 1e-12, 1.0)
    return -np.sum(y_true * np.log(y_pred_safe))


# ─────────────────────────────────────────────
# 1. MODÈLE UNIGRAMME
# ─────────────────────────────────────────────

def train_unigram(corpus: list) -> dict:
    """
    Entraîne un modèle de langue unigramme sur un corpus de tokens.

    Le modèle apprend la probabilité de chaque mot :
        P(w) = nb_occurrences(w) / total_mots

    Paramètre :
        corpus (list) : liste de tokens (mots)

    Retour :
        dict : {mot: probabilité}

    Exemple :
        >>> corpus = ["le", "chat", "dort", "le", "chat", "mange"]
        >>> train_unigram(corpus)
        {"le": 0.333, "chat": 0.333, "dort": 0.167, "mange": 0.167}
    """
    total = len(corpus)

    # Compter les occurrences de chaque mot
    comptages = Counter(corpus)

    # Calculer les probabilités
    modele = {mot: count / total for mot, count in comptages.items()}
    return modele


def unigram_probability(modele: dict, sequence: list,
                        smoothing: float = 1e-6) -> float:
    """
    Calcule la probabilité d'une séquence sous un modèle unigramme.

    P(w1, w2, ..., wn) = P(w1) × P(w2) × ... × P(wn)

    Paramètres :
        modele (dict)    : modèle unigramme entraîné {mot: proba}
        sequence (list)  : liste de tokens à évaluer
        smoothing (float): probabilité pour les mots hors vocabulaire

    Retour :
        float : probabilité de la séquence

    Note : Les probabilités s'accumulent par multiplication.
           Pour des séquences longues, utiliser log-probabilités
           pour éviter les underflow numériques.
    """
    proba = 1.0
    for mot in sequence:
        # Si le mot n'est pas dans le vocabulaire → smoothing
        p_mot = modele.get(mot, smoothing)
        proba *= p_mot
    return proba


# ─────────────────────────────────────────────
# 2. MODÈLE BIGRAMME
# ─────────────────────────────────────────────

def train_bigram(corpus: list) -> dict:
    """
    Entraîne un modèle bigramme sur un corpus.

    Le modèle apprend les probabilités conditionnelles :
        P(wi | wi-1) = C(wi-1, wi) / C(wi-1)

    Un token spécial <START> marque le début de chaque phrase.

    Paramètre :
        corpus (list) : liste de tokens

    Retour :
        dict : {mot_precedent: {mot_suivant: probabilité_conditionnelle}}

    Exemple :
        >>> corpus = ["<START>", "le", "chat", "dort", "<START>", "le", "chien", "court"]
        Bigrammes extraits : (<START>, le), (le, chat), (chat, dort), etc.
    """
    # Ajouter <START> au début du corpus pour le premier mot
    corpus_avec_start = ["<START>"] + corpus

    # Compter les bigrammes (paires de mots consécutifs)
    comptages_bigrammes = Counter()
    comptages_unigrammes = Counter()

    for i in range(len(corpus_avec_start) - 1):
        w_prec = corpus_avec_start[i]       # Mot précédent
        w_suiv = corpus_avec_start[i + 1]   # Mot suivant

        comptages_bigrammes[(w_prec, w_suiv)] += 1
        comptages_unigrammes[w_prec] += 1

    # Calculer les probabilités conditionnelles
    # P(w_suiv | w_prec) = C(w_prec, w_suiv) / C(w_prec)
    modele = {}
    for (w_prec, w_suiv), count in comptages_bigrammes.items():
        if w_prec not in modele:
            modele[w_prec] = {}
        modele[w_prec][w_suiv] = count / comptages_unigrammes[w_prec]

    return modele


def bigram_probability(modele_bigram: dict, modele_unigram: dict,
                       sequence: list, smoothing: float = 1e-6) -> float:
    """
    Calcule la probabilité d'une séquence sous un modèle bigramme.

    P(w1, ..., wn) = P(w1 | <START>) × P(w2 | w1) × ... × P(wn | wn-1)

    Paramètres :
        modele_bigram (dict)  : {mot_prec: {mot_suiv: proba}}
        modele_unigram (dict) : pour le premier mot {mot: proba}
        sequence (list)       : liste de tokens à évaluer
        smoothing (float)     : probabilité pour les bigrammes inconnus

    Retour :
        float : probabilité de la séquence
    """
    if not sequence:
        return 1.0

    proba = 1.0
    mot_prec = "<START>"  # Contexte initial

    for mot in sequence:
        # P(mot | mot_prec) depuis le modèle bigramme
        if mot_prec in modele_bigram and mot in modele_bigram[mot_prec]:
            p = modele_bigram[mot_prec][mot]
        else:
            p = smoothing  # Bigramme jamais vu → probabilité minimale

        proba *= p
        mot_prec = mot  # Mettre à jour le contexte

    return proba


# ─────────────────────────────────────────────
# 3. PERPLEXITÉ
# ─────────────────────────────────────────────

def perplexity_unigram(modele: dict, sequences_test: list,
                       smoothing: float = 1e-6) -> float:
    """
    Calcule la perplexité d'un modèle unigramme sur des séquences de test.

    PPL = exp( -1/N × Σ log P(wi) )

    Paramètres :
        modele (dict)         : modèle unigramme entraîné
        sequences_test (list) : liste de listes de tokens
        smoothing (float)     : probabilité pour mots inconnus

    Retour :
        float : perplexité (plus basse = meilleur modèle)

    Interprétation :
        PPL = k signifie que le modèle est aussi incertain
        qu'un choix uniforme parmi k mots à chaque position.
    """
    log_proba_total = 0.0
    N = 0  # Nombre total de tokens

    for sequence in sequences_test:
        for mot in sequence:
            p = modele.get(mot, smoothing)
            log_proba_total += math.log(p)
            N += 1

    if N == 0:
        return float('inf')

    # Perplexité = exp de la log-probabilité moyenne négative
    return math.exp(-log_proba_total / N)


# ─────────────────────────────────────────────
# 4. EXEMPLE COMPLET ET COMPARAISON
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Softmax ──
    print("=== Test Softmax ===")
    scores = np.array([2.0, 1.0, 0.1])
    probas = softmax(scores)
    print(f"Scores  : {scores}")
    print(f"Softmax : {probas.round(3)}")
    print(f"Somme   : {probas.sum():.6f}")  # Doit être ≈ 1.0

    # ── Cross-entropie ──
    print("\n=== Test Cross-entropie ===")
    y_true = np.array([0, 1, 0])
    y_pred_bon  = np.array([0.1, 0.8, 0.1])
    y_pred_mauv = np.array([0.4, 0.2, 0.4])
    print(f"Bonne prédiction : CE = {cross_entropy(y_true, y_pred_bon):.3f}")
    print(f"Mauvaise prédiction : CE = {cross_entropy(y_true, y_pred_mauv):.3f}")

    # ── Modèles de langue ──
    print("\n=== Modèles de langue ===")
    corpus = ["le", "chat", "dort", "le", "chat", "mange", "le", "poisson",
              "le", "chien", "court", "le", "chien", "aboie"]

    # Entraîner les deux modèles
    uni = train_unigram(corpus)
    bi  = train_bigram(corpus)

    print("\nProbabilités unigramme (top 5) :")
    for mot, p in sorted(uni.items(), key=lambda x: -x[1])[:5]:
        print(f"  P({mot}) = {p:.3f}")

    # Tester sur une séquence
    sequence_test = ["le", "chat", "dort"]
    p_uni = unigram_probability(uni, sequence_test)
    p_bi  = bigram_probability(bi, uni, sequence_test)

    print(f"\nSéquence : {sequence_test}")
    print(f"  P(unigramme) = {p_uni:.6f}")
    print(f"  P(bigramme)  = {p_bi:.6f}")
    print(f"  → Le bigramme est {'plus' if p_bi > p_uni else 'moins'} élevé")
    print(f"    (il tire parti des régularités du corpus)")

    # Perplexité
    test_seqs = [["le", "chat"], ["le", "chien"]]
    ppl = perplexity_unigram(uni, test_seqs)
    print(f"\nPerplexité unigramme sur séquences test : {ppl:.2f}")
