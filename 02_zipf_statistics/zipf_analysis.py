"""
=============================================================
02 — Statistiques textuelles et loi de Zipf
=============================================================

CONCEPT CLÉ — LOI DE ZIPF :
    En 1935, George Kingsley Zipf observe que dans tout texte
    naturel, la fréquence d'un mot est inversement proportionnelle
    à son rang :
        f(rang) ∝ 1 / rang

    Autrement dit :
        - Le mot le plus fréquent apparaît ~2× plus que le 2e
        - Le 2e apparaît ~2× plus que le 3e, etc.

    Sur un graphe log-log, cela donne une droite.
    → Universel dans TOUTES les langues naturelles.

GÉNÉRALISATION :
    Fréquence des mots → trier → tracer en log-log
    Si la courbe est une droite : la loi de Zipf est vérifiée.

APPLICATIONS :
    - Identifier la langue d'un texte
    - Détecter les mots outils (très fréquents)
    - Compression de texte
    - Modèles de langue probabilistes
=============================================================
"""

import matplotlib.pyplot as plt
from collections import Counter


# ─────────────────────────────────────────────
# 1. COMPTAGE DES FRÉQUENCES
# ─────────────────────────────────────────────

def compter_occurrences(liste_mots: list) -> dict:
    """
    Compte combien de fois chaque mot apparaît dans une liste de tokens.

    Paramètre :
        liste_mots (list) : liste de tokens (mots)

    Retour :
        dict : dictionnaire {mot: fréquence}

    Exemple :
        >>> compter_occurrences(["le", "chat", "le", "chat", "dort"])
        {'le': 2, 'chat': 2, 'dort': 1}

    Alternative avec Counter (plus concise) :
        from collections import Counter
        Counter(liste_mots)
    """
    dic = {}
    for mot in liste_mots:
        if mot not in dic:
            # Premier occurrence : initialiser à 1
            dic[mot] = 1
        else:
            # Occurrences suivantes : incrémenter
            dic[mot] += 1
    return dic


def compter_longueurs(liste_mots: list) -> dict:
    """
    Compte combien de mots ont chaque longueur (en caractères).

    Paramètre :
        liste_mots (list) : liste de tokens

    Retour :
        dict : dictionnaire {longueur: nombre_de_mots}

    Exemple :
        >>> compter_longueurs(["le", "chat", "dort"])
        {2: 1, 4: 2}   # 1 mot de 2 lettres, 2 mots de 4 lettres
    """
    dic = {}
    for mot in liste_mots:
        longueur = len(mot)
        if longueur not in dic:
            dic[longueur] = 1
        else:
            dic[longueur] = dic[longueur] + 1
    return dic


# ─────────────────────────────────────────────
# 2. VISUALISATION — LOI DE ZIPF
# ─────────────────────────────────────────────

def graph_zipf(dic_occurrences: dict, langue: str) -> None:
    """
    Trace la courbe de Zipf pour un dictionnaire de fréquences.

    La loi de Zipf se visualise en triant les fréquences par ordre
    décroissant et en traçant le graphe rang/fréquence en échelle
    logarithmique. Une ligne droite confirme la loi.

    Paramètres :
        dic_occurrences (dict) : dictionnaire {mot: fréquence}
        langue (str)           : nom de la langue (pour la légende)

    Usage typique :
        # Tracer plusieurs langues sur le même graphe
        graph_zipf(freq_fr, "Français")
        graph_zipf(freq_en, "Anglais")
        plt.legend()
        plt.show()
    """
    # Trier les fréquences du plus grand au plus petit (rang 1 = mot le plus fréquent)
    liste_effectifs = sorted(dic_occurrences.values(), reverse=True)

    plt.plot(liste_effectifs, label=langue)
    plt.xlabel("Rang du mot")
    plt.ylabel("Fréquence")
    plt.title("Loi de Zipf")

    # IMPORTANT : les deux axes en log pour obtenir une droite
    plt.xscale("log")
    plt.yscale("log")


def graph_longueurs(dic_longueurs: dict, label: str, total_mots: int) -> None:
    """
    Trace la distribution des longueurs de mots (normalisée).

    La courbe montre la proportion de mots ayant chaque longueur
    (entre 0 et 30 caractères). Chaque langue a une distribution
    caractéristique.

    Paramètres :
        dic_longueurs (dict) : dictionnaire {longueur: nombre_mots}
        label (str)          : étiquette pour la légende
        total_mots (int)     : nombre total de mots (pour normaliser)
    """
    liste_effectifs = []
    # On crée une liste de 0 à 29 (longueurs possibles de 0 à 29)
    for longueur in range(30):
        if longueur in dic_longueurs:
            # Normalisation : proportion = nb_mots_de_cette_longueur / total_mots
            liste_effectifs.append(dic_longueurs[longueur] / total_mots)
        else:
            # Cette longueur n'apparaît pas dans le corpus → 0
            liste_effectifs.append(0)

    plt.plot(liste_effectifs, label=label)
    plt.xlabel("Longueur du mot")
    plt.ylabel("Effectif (proportion)")
    plt.title("Distribution des longueurs de mots")


# ─────────────────────────────────────────────
# 3. EXEMPLE COMPLET D'UTILISATION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulation d'un petit corpus
    texte = """
    le chat mange le poisson le chien court dans le jardin
    la souris mange du fromage la vache broute l herbe
    le lapin saute dans le pré le chat dort sur le canapé
    """
    tokens = texte.split()

    # ── Comptage ──
    freq = compter_occurrences(tokens)
    print("Fréquences (extrait) :")
    # Afficher les 5 mots les plus fréquents
    top5 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
    for mot, nb in top5:
        print(f"  '{mot}' : {nb} fois")

    # ── Longueurs ──
    long = compter_longueurs(tokens)
    print("\nDistribution des longueurs :")
    for l, nb in sorted(long.items()):
        print(f"  {l} lettres : {nb} mot(s)")

    # ── Graphiques ──
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    graph_zipf(freq, "Corpus exemple")
    plt.legend()

    plt.subplot(1, 2, 2)
    graph_longueurs(long, "Corpus exemple", len(tokens))
    plt.legend()

    plt.tight_layout()
    plt.savefig("zipf_exemple.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("\nGraphique sauvegardé → zipf_exemple.png")
