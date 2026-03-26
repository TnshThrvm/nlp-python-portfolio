"""
=============================================================
01 — Lecture et nettoyage de fichiers texte
=============================================================

CONCEPT CLÉ :
    Avant toute analyse NLP, il faut lire et nettoyer les données.
    Ce module regroupe les fonctions de base pour charger du texte
    depuis différents formats (TXT, HTML, JSON).

GÉNÉRALISATION :
    Toute pipeline NLP commence par ces étapes :
        Fichier brut → Lecture → Texte propre → Tokenisation → Analyse

FORMATS SUPPORTÉS :
    - .txt  → lecture directe
    - .html → extraction du texte via BeautifulSoup
    - .json → chargement structuré
=============================================================
"""

import json
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────
# 1. LECTURE DE FICHIERS TEXTE BRUT
# ─────────────────────────────────────────────

def lire_fichier(chemin: str) -> str:
    """
    Lit un fichier texte (.txt) et retourne son contenu sous forme de chaîne.

    Paramètre :
        chemin (str) : chemin vers le fichier à lire

    Retour :
        str : le contenu complet du fichier

    Exemple :
        texte = lire_fichier("corpus/mon_texte.txt")
        print(texte[:100])   # affiche les 100 premiers caractères
    """
    with open(chemin, 'r', encoding='utf-8') as f:
        texte = f.read()
    return texte


def ouvrir(chemin: str) -> str:
    """
    Alias de lire_fichier. Même comportement, nom différent.
    Utilisé dans certains notebooks pour plus de lisibilité.
    """
    with open(chemin, encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────
# 2. LECTURE DE FICHIERS HTML
# ─────────────────────────────────────────────

def lire_fichier_html(chemin: str) -> str:
    """
    Lit un fichier HTML et retourne uniquement le texte visible,
    sans les balises HTML.

    Utilise BeautifulSoup pour parser le HTML et extraire
    le texte brut avec get_text().

    Paramètre :
        chemin (str) : chemin vers le fichier HTML

    Retour :
        str : texte brut extrait du HTML

    Exemple :
        texte = lire_fichier_html("pages/article.html")
        # Le texte ne contient plus de balises <p>, <div>, etc.
    """
    with open(chemin, 'r', encoding='utf-8') as f:
        # Crée un objet BeautifulSoup pour analyser le HTML
        soup = BeautifulSoup(f, "html.parser")
        # get_text() extrait tout le texte visible, sans les balises
        texte = soup.get_text()
    return texte


# ─────────────────────────────────────────────
# 3. TOKENISATION (DÉCOUPAGE EN MOTS)
# ─────────────────────────────────────────────

def couper(chaine: str) -> list:
    """
    Découpe une chaîne de caractères en liste de tokens (mots).
    Utilise le découpage par espaces (split).

    Paramètre :
        chaine (str) : texte à découper

    Retour :
        list : liste de tokens

    Exemple :
        >>> couper("le chat dort")
        ['le', 'chat', 'dort']

    Note : Cette tokenisation est simpliste (split sur les espaces).
           Pour un traitement plus fin, utiliser SpaCy ou NLTK.
    """
    return chaine.split()


# ─────────────────────────────────────────────
# 4. LECTURE ET ÉCRITURE JSON
# ─────────────────────────────────────────────

def charger_json(nom_fichier: str) -> dict:
    """
    Charge un fichier JSON et retourne son contenu sous forme
    de dictionnaire Python.

    Paramètre :
        nom_fichier (str) : chemin vers le fichier JSON

    Retour :
        dict ou list : données désérialisées depuis le JSON

    Exemple :
        donnees = charger_json("entites.json")
        print(donnees["entité_0"])
    """
    with open(nom_fichier, "r", encoding='utf-8') as f:
        return json.load(f)


def sauvegarder_json(dic: dict, nom_fichier: str) -> None:
    """
    Sauvegarde un dictionnaire Python dans un fichier JSON.
    Le paramètre indent=2 rend le fichier lisible par un humain.

    Paramètres :
        dic (dict)        : données à sauvegarder
        nom_fichier (str) : chemin du fichier de sortie

    Exemple :
        resultats = {"entité_0": {"Entité": "Paris", "Label": "LOC"}}
        sauvegarder_json(resultats, "resultats.json")
    """
    with open(nom_fichier, "w", encoding='utf-8') as w:
        # indent=2 : indentation de 2 espaces pour la lisibilité
        json.dump(dic, w, indent=2)


# Alias avec un nom différent (les deux font la même chose)
enregistrer_json = sauvegarder_json


# ─────────────────────────────────────────────
# 5. EXEMPLE D'UTILISATION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Démonstration sur un texte fictif ──

    # Simulation d'un texte (en pratique, on lirait un vrai fichier)
    texte_exemple = "Le chat dort. La souris mange du fromage. Le chien aboie."

    # Tokenisation
    tokens = couper(texte_exemple)
    print("Tokens :", tokens)
    # → ['Le', 'chat', 'dort.', 'La', 'souris', ...]

    # Nombre de mots
    print("Nombre de mots :", len(tokens))

    # Sauvegarde et rechargement JSON
    donnees = {"texte": texte_exemple, "nb_tokens": len(tokens)}
    sauvegarder_json(donnees, "exemple_sortie.json")

    rechargement = charger_json("exemple_sortie.json")
    print("Données rechargées :", rechargement)

    # Nettoyage du fichier temporaire
    import os
    os.remove("exemple_sortie.json")
    print("Démonstration terminée ✓")
