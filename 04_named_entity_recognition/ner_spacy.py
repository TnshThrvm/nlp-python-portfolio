"""
=============================================================
04 — Reconnaissance d'entités nommées (NER) avec SpaCy
=============================================================

CONCEPT CLÉ — NER (Named Entity Recognition) :
    La NER consiste à identifier et classer automatiquement des
    "entités nommées" dans un texte :

    TYPES D'ENTITÉS (labels SpaCy) :
        PER  → Personnes          ("Marie Curie", "Victor Hugo")
        LOC  → Lieux              ("Paris", "Alpes", "Seine")
        ORG  → Organisations      ("ONU", "Sorbonne", "SNCF")
        MISC → Divers             ("français", "Nobel")

    COMMENT ÇA MARCHE :
        Texte brut
            ↓ Tokenisation
        ["Marie", "Curie", "est", "née", "en", "Pologne"]
            ↓ Modèle de langue (réseau de neurones entraîné)
        [PER: "Marie Curie", LOC: "Pologne"]

MODÈLES SPACY FRANÇAIS :
    fr_core_news_sm  → petit modèle, rapide, moins précis
    fr_core_news_md  → modèle moyen (équilibre vitesse/précision)
    fr_core_news_lg  → grand modèle, lent mais plus précis

GÉNÉRALISATION :
    SpaCy suit le même pipeline pour toutes les langues :
        nlp = spacy.load("modele")
        doc = nlp(texte)
        for entite in doc.ents:
            print(entite.text, entite.label_)
=============================================================
"""

import json
import glob
import time
import spacy


# ─────────────────────────────────────────────
# 1. EXTRACTION DES ENTITÉS NOMMÉES
# ─────────────────────────────────────────────

def extraire_entites(texte: str, nlp, limite_chars: int = 5000) -> dict:
    """
    Extrait les entités nommées d'un texte avec SpaCy.

    Paramètres :
        texte (str)        : texte à analyser
        nlp                : modèle SpaCy chargé (spacy.load("..."))
        limite_chars (int) : nombre max de caractères à traiter
                             (évite les délais sur les très longs textes)

    Retour :
        dict : {
            "entité_0": {"Entité": "Paris", "Label": "LOC"},
            "entité_1": {"Entité": "Victor Hugo", "Label": "PER"},
            ...
        }

    Exemple :
        nlp = spacy.load("fr_core_news_sm")
        entites = extraire_entites("Victor Hugo est né à Besançon.", nlp)
        # → {"entité_0": {"Entité": "Victor Hugo", "Label": "PER"},
        #    "entité_1": {"Entité": "Besançon", "Label": "LOC"}}
    """
    # On limite la taille pour le traitement (les modèles ont une limite)
    doc = nlp(texte[:limite_chars])

    # Construire le dictionnaire des entités détectées
    dic_entites = {}
    for i, entite in enumerate(doc.ents):
        dic_entites[f"entité_{i}"] = {
            "Entité": entite.text,   # Le texte de l'entité
            "Label": entite.label_   # Sa catégorie (PER, LOC, ORG, MISC)
        }

    return dic_entites


def traiter_corpus_ner(liste_fichiers: list, nom_modele: str = "fr_core_news_sm") -> None:
    """
    Applique la NER à tous les fichiers d'un corpus et sauvegarde
    les résultats en JSON.

    Paramètres :
        liste_fichiers (list) : liste de chemins vers les fichiers .txt
        nom_modele (str)      : nom du modèle SpaCy à utiliser

    Fichiers créés :
        entites_[nom_modele].json pour chaque fichier traité

    Exemple d'utilisation :
        fichiers = glob.glob("corpus/*/*.txt")
        traiter_corpus_ner(fichiers, "fr_core_news_sm")
    """
    # Charger le modèle UNE SEULE FOIS (opération coûteuse)
    print(f"Chargement du modèle : {nom_modele}...")
    nlp = spacy.load(nom_modele)

    for fichier in liste_fichiers:
        print(f"\nFichier : {fichier}")

        # Lire le texte
        with open(fichier, 'r', encoding='utf-8') as f:
            texte = f.read()
        print(f"  Taille : {len(texte)} caractères")

        # Mesurer le temps de traitement
        debut = time.time()

        # Extraire les entités
        entites = extraire_entites(texte, nlp)

        duree = time.time() - debut

        # Sauvegarder les résultats
        nom_sortie = f"entites_{nom_modele}.json"
        with open(nom_sortie, "w", encoding="utf-8") as f:
            json.dump(entites, f, indent=2, ensure_ascii=False)

        print(f"  Entités détectées : {len(entites)}")
        print(f"  Temps de traitement : {duree:.2f}s")
        print(f"  Résultats sauvegardés → {nom_sortie}")


# ─────────────────────────────────────────────
# 2. ÉVALUATION SIMPLE (VP / FP / FN)
# ─────────────────────────────────────────────

def lire_annotations_csv(chemin_csv: str) -> set:
    """
    Lit un fichier CSV d'annotations humaines et retourne
    l'ensemble des mots annotés comme entités.

    Format CSV attendu :
        mot;PER;LOC;ORG;MISC
        Victor;X;;;
        Hugo;X;;;
        est;;;;
        né;;;;
        en;;;;
        Pologne;;X;;

    Paramètre :
        chemin_csv (str) : chemin vers le fichier CSV

    Retour :
        set : ensemble des mots annotés comme entités
              (au moins une colonne marquée 'X')

    Note : La présence d'un 'X' dans n'importe quelle colonne
           d'entité signifie que ce mot fait partie d'une entité.
    """
    annotations = set()

    with open(chemin_csv, "r", encoding="utf-8") as f:
        lignes = f.readlines()

    # Ignorer la première ligne (en-tête)
    for ligne in lignes[1:]:
        ligne = ligne.strip()
        if not ligne:
            continue  # Ignorer les lignes vides

        colonnes = ligne.split(";")
        mot = colonnes[0].strip()  # La première colonne = le mot

        # Si au moins une colonne (hors la première) contient 'X'
        # → ce mot est annoté comme entité
        if any("X" in cellule for cellule in colonnes[1:]):
            annotations.add(mot)

    return annotations


def evaluer_ner(texte: str, annotations_humaines: set, nlp) -> dict:
    """
    Évalue les performances de SpaCy en comparant ses prédictions
    avec des annotations humaines (vérité terrain).

    Paramètres :
        texte (str)                : texte à analyser
        annotations_humaines (set) : entités annotées par un humain
        nlp                        : modèle SpaCy

    Retour :
        dict : {"VP": int, "FP": int, "FN": int,
                "VP_liste": set, "FP_liste": set, "FN_liste": set}

    Définitions :
        VP (Vrai Positif)  : détecté par SpaCy ET dans les annotations humaines
        FP (Faux Positif)  : détecté par SpaCy MAIS pas dans les annotations humaines
        FN (Faux Négatif)  : dans les annotations humaines MAIS pas détecté par SpaCy
    """
    doc = nlp(texte[:5000])

    # Entités détectées par SpaCy (ensemble pour éviter les doublons)
    entites_spacy = set(entite.text for entite in doc.ents)

    # Calcul des VP, FP, FN par opérations d'ensembles
    VP = entites_spacy & annotations_humaines   # intersection
    FP = entites_spacy - annotations_humaines   # dans SpaCy, pas dans humain
    FN = annotations_humaines - entites_spacy   # dans humain, pas dans SpaCy

    print(f"VP (corrects) : {len(VP)}")
    print(f"FP (faux positifs SpaCy) : {len(FP)}")
    print(f"FN (manqués par SpaCy) : {len(FN)}")

    return {
        "VP": len(VP), "FP": len(FP), "FN": len(FN),
        "VP_liste": VP, "FP_liste": FP, "FN_liste": FN
    }


# ─────────────────────────────────────────────
# 3. EXEMPLE D'UTILISATION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Exemple sans fichiers réels (démonstration) ──

    texte_demo = """
    Victor Hugo est né à Besançon le 26 février 1802.
    Il est l'auteur des Misérables et de Notre-Dame de Paris.
    Il a vécu à Paris, à Guernesey, et est mort en France en 1885.
    L'Académie française lui a rendu hommage.
    """

    print("=== Démonstration NER avec SpaCy ===\n")

    # Charger le modèle (décommenter si SpaCy est installé)
    # nlp = spacy.load("fr_core_news_sm")
    # entites = extraire_entites(texte_demo, nlp)
    # for cle, val in entites.items():
    #     print(f"  {val['Label']:4} | {val['Entité']}")

    # Résultat attendu :
    print("Résultat attendu :")
    print("  PER  | Victor Hugo")
    print("  LOC  | Besançon")
    print("  MISC | Misérables")
    print("  LOC  | Notre-Dame de Paris")
    print("  LOC  | Paris")
    print("  LOC  | Guernesey")
    print("  LOC  | France")
    print("  ORG  | Académie française")

    print("\nPour utiliser sur de vrais fichiers :")
    print("  import glob, spacy")
    print("  fichiers = glob.glob('corpus/*/*.txt')")
    print("  traiter_corpus_ner(fichiers, 'fr_core_news_sm')")
