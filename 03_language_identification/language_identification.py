"""
=============================================================
03 — Identification de langue par mots fréquents
=============================================================

CONCEPT CLÉ — MODÈLE DE LANGUE PAR LISTE DE MOTS :
    Principe : chaque langue a ses propres mots très fréquents
    (articles, prépositions, conjonctions = mots "outils").
        Français : le, la, les, de, du, un, une, et, est...
        Anglais  : the, a, of, and, is, in, to, it...
        Espagnol : el, la, los, de, y, en, que, un...

    ALGORITHME :
        1. ENTRAÎNEMENT : lire des corpus par langue
           → extraire les N mots les plus fréquents de chaque langue
           → créer un dictionnaire {langue: [top_mots]}

        2. TEST : lire un texte inconnu
           → extraire ses N mots les plus fréquents
           → comparer avec chaque langue (intersection)
           → la langue avec le plus grand chevauchement = langue détectée

GÉNÉRALISATION :
    C'est un classifieur naïf basé sur les fréquences.
    Plus N est grand, meilleure est la discrimination entre langues.
    N = 10 à 50 mots suffit souvent pour discriminer les langues européennes.

STRUCTURE DES DOSSIERS ATTENDUE :
    corpus/
    ├── fr/
    │   ├── texte1.txt
    │   └── texte2.txt
    ├── en/
    │   └── texte1.txt
    └── es/
        └── texte1.txt
=============================================================
"""

import glob
import os
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────
# 1. CONSTRUCTION DU MODÈLE DE LANGUE
# ─────────────────────────────────────────────

def lire_fichier(chemin: str) -> str:
    """Lit un fichier texte brut (encodage UTF-8)."""
    with open(chemin, 'r', encoding='utf-8') as f:
        return f.read()


def lire_fichier_html(chemin: str) -> str:
    """Lit un fichier HTML et extrait le texte brut."""
    with open(chemin, 'r', encoding='utf-8') as f:
        return BeautifulSoup(f, "html.parser").get_text()


def compter_occurrences(liste_mots: list) -> dict:
    """Compte la fréquence de chaque mot dans une liste."""
    dic = {}
    for mot in liste_mots:
        dic[mot] = dic.get(mot, 0) + 1
    return dic


def traiter_fichiers(motif: str) -> dict:
    """
    Parcourt tous les fichiers correspondant au motif glob,
    déduit la langue depuis le nom du dossier parent,
    et construit un dictionnaire de fréquences par langue.

    Paramètre :
        motif (str) : motif glob pour trouver les fichiers
                      Exemple : "corpus/*/*.*"
                      Structure attendue : corpus/[LANGUE]/fichier.txt

    Retour :
        dict : {langue: {mot: fréquence_totale_dans_tous_les_fichiers_de_cette_langue}}

    Exemple de sortie :
        {
            'fr': {'le': 1200, 'de': 890, 'la': 780, ...},
            'en': {'the': 1500, 'of': 930, 'and': 870, ...}
        }
    """
    dic_langue = {}

    for chemin in glob.glob(motif):
        # Normaliser le séparateur (Windows/Linux/Mac)
        # Exemple : "corpus\\fr\\texte.txt" → "corpus/fr/texte.txt"
        chemin_normalise = chemin.replace('\\', '/')
        dossiers = chemin_normalise.split('/')

        # Le nom de la langue = nom du dossier parent du fichier
        # Ex : ["corpus", "fr", "texte.txt"] → langue = "fr" (index 1)
        try:
            langue = dossiers[1]
        except IndexError:
            langue = "inconnu"

        # Initialiser le dictionnaire de cette langue s'il n'existe pas encore
        if langue not in dic_langue:
            dic_langue[langue] = {}

        # Choisir la bonne fonction de lecture selon l'extension
        if chemin.endswith('.html'):
            texte = lire_fichier_html(chemin)
        else:
            texte = lire_fichier(chemin)

        # Compter les mots du fichier
        liste_mots = texte.split()
        frequences_fichier = compter_occurrences(liste_mots)

        # Ajouter au dictionnaire global de la langue (accumulation)
        for mot, freq in frequences_fichier.items():
            dic_langue[langue][mot] = dic_langue[langue].get(mot, 0) + freq

    return dic_langue


# ─────────────────────────────────────────────
# 2. EXTRACTION DES MOTS LES PLUS FRÉQUENTS
# ─────────────────────────────────────────────

def extraire_top_mots(dic_langue: dict, n: int = 10) -> dict:
    """
    Pour chaque langue, extrait les N mots les plus fréquents.
    Ces mots serviront de "signature" de la langue.

    Paramètres :
        dic_langue (dict) : {langue: {mot: fréquence}}
        n (int)           : nombre de mots à extraire (défaut : 10)

    Retour :
        dict : {langue: [mot_1, mot_2, ..., mot_n]}
               (du moins fréquent au plus fréquent dans la liste)

    Exemple :
        >>> extraire_top_mots({'fr': {'le': 100, 'de': 80, 'la': 60}}, n=2)
        {'fr': ['la', 'le']}
        # 'la' et 'le' sont les 2 mots les + fréquents
    """
    dic_top = {}
    for langue, dic_effectifs in dic_langue.items():
        # Créer des paires (fréquence, mot) pour pouvoir trier par fréquence
        liste_freq_mot = [(freq, mot) for mot, freq in dic_effectifs.items()]

        # Trier par fréquence croissante
        liste_freq_mot.sort()

        # Prendre les N derniers (= les N plus fréquents)
        top_n_mots = [mot for freq, mot in liste_freq_mot[-n:]]
        dic_top[langue] = top_n_mots

    return dic_top


# ─────────────────────────────────────────────
# 3. COMPARAISON ET IDENTIFICATION DE LANGUE
# ─────────────────────────────────────────────

def comparer_modeles(modele: dict, test: dict) -> dict:
    """
    Compare deux modèles (train vs test) en calculant l'intersection
    de leurs listes de mots fréquents.

    Plus l'intersection est grande, plus le texte test ressemble
    à cette langue.

    Paramètres :
        modele (dict) : {langue: [top_mots_train]}  ← modèle entraîné
        test (dict)   : {langue: [top_mots_test]}   ← texte à identifier

    Retour :
        dict : {langue: [mots_en_commun]}

    Exemple :
        modele = {'fr': ['le', 'de', 'la'], 'en': ['the', 'of', 'and']}
        test   = {'inconnu': ['le', 'de', 'un']}
        # Intersection avec 'fr' = ['le', 'de'] → score 2
        # Intersection avec 'en' = []            → score 0
        # → Le texte est probablement en français
    """
    dic_intersection = {}
    for langue in modele:
        if langue in test:
            # Intersection = mots présents dans les deux listes
            intersection = list(set(modele[langue]) & set(test[langue]))
            dic_intersection[langue] = intersection
    return dic_intersection


def identifier_langue(texte_inconnu: str, modele: dict, n: int = 20) -> str:
    """
    Identifie la langue d'un texte inconnu en le comparant au modèle.

    Paramètres :
        texte_inconnu (str) : texte dont on veut identifier la langue
        modele (dict)       : {langue: [top_mots]} — modèle entraîné
        n (int)             : taille du top à comparer

    Retour :
        str : langue identifiée

    Exemple :
        langue = identifier_langue("The cat is on the mat", modele_entraine)
        print(langue)   # → 'en'
    """
    # Construire le profil du texte inconnu
    tokens = texte_inconnu.split()
    freq_inconnu = compter_occurrences(tokens)

    # Extraire les top mots du texte inconnu
    liste = [(f, m) for m, f in freq_inconnu.items()]
    liste.sort()
    top_inconnu = [m for f, m in liste[-n:]]

    # Comparer avec chaque langue du modèle
    scores = {}
    for langue, top_langue in modele.items():
        # Score = taille de l'intersection
        intersection = set(top_langue) & set(top_inconnu)
        scores[langue] = len(intersection)

    # Retourner la langue avec le score le plus élevé
    return max(scores, key=scores.get)


# ─────────────────────────────────────────────
# 4. EXEMPLE COMPLET
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Simulation sans fichiers réels ──
    # En pratique, utiliser traiter_fichiers("corpus/*/*.txt")

    # Simulation d'un mini-modèle de langue
    modele_langue = {
        'fr': ['le', 'de', 'la', 'et', 'les', 'des', 'en', 'du', 'un', 'une',
               'est', 'que', 'qui', 'dans', 'il', 'pas', 'par', 'sur', 'au', 'se'],
        'en': ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'that', 'it', 'was',
               'for', 'on', 'are', 'as', 'with', 'his', 'they', 'at', 'be', 'this'],
        'es': ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se',
               'las', 'un', 'por', 'con', 'no', 'una', 'su', 'para', 'es', 'al']
    }

    # Textes à tester
    textes_test = [
        ("Le chat dort sur le canapé et la souris mange du fromage.", "fr"),
        ("The quick brown fox jumps over the lazy dog in the park.", "en"),
        ("El gato está en la casa y el perro corre en el jardín.", "es"),
    ]

    print("=== Test d'identification de langue ===\n")
    for texte, vraie_langue in textes_test:
        langue_predite = identifier_langue(texte, modele_langue, n=10)
        correct = "✓" if langue_predite == vraie_langue else "✗"
        print(f"{correct} Prédit: {langue_predite} | Attendu: {vraie_langue}")
        print(f"  Texte : '{texte[:50]}...'")
        print()
