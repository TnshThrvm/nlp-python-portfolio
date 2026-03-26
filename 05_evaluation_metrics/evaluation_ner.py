"""
=============================================================
05 — Évaluation de systèmes NER : Précision, Rappel, F1
=============================================================

CONCEPT CLÉ — MÉTRIQUES D'ÉVALUATION :
    Pour mesurer les performances d'un système automatique,
    on compare ses prédictions à une "vérité terrain" (ground truth)
    annotée par des humains.

    MATRICE DE CONFUSION (pour un token) :

                        | Système dit OUI | Système dit NON
        ─────────────────────────────────────────────────────
        Humain dit OUI  |   VP (Vrai +)   |  FN (Faux -)
        Humain dit NON  |   FP (Faux +)   |  VN (Vrai -)

    MÉTRIQUES :
        Précision = VP / (VP + FP)   ← "Quand il dit oui, a-t-il raison ?"
        Rappel    = VP / (VP + FN)   ← "Trouve-t-il tout ce qui existe ?"
        F1        = 2 × P × R / (P + R)  ← Moyenne harmonique (équilibre P et R)

    EXEMPLE CONCRET :
        Sur 10 entités réelles :
        - Système en trouve 7 correctement (VP=7)
        - En invente 3 qui n'existent pas (FP=3)
        - En rate 3 (FN=3)
        → Précision = 7/(7+3) = 0.70  → 70% de ce qu'il dit est correct
        → Rappel    = 7/(7+3) = 0.70  → Il trouve 70% de ce qui existe
        → F1        = 0.70

GÉNÉRALISATION :
    Ces métriques s'appliquent à tout système de classification binaire :
    détection de spam, traduction, résumé automatique, etc.

FORMAT CSV ATTENDU :
    token;PER;LOC;ORG;MISC
    Victor;X;;;
    Hugo;X;;;
    est;;;;
    né;;;;
    en;;;;
    Paris;;X;;
=============================================================
"""

import re
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1. LECTURE DES FICHIERS D'ANNOTATION
# ─────────────────────────────────────────────

def lire_annotations_csv(chemin_csv: str) -> list:
    """
    Lit un fichier CSV d'annotations NER et retourne une liste
    de tuples (token, dictionnaire_d_annotations).

    Paramètre :
        chemin_csv (str) : chemin vers le fichier CSV

    Retour :
        list : [(token, {"PER": "X" ou "", "LOC": "...", ...}), ...]

    Exemple de sortie :
        [
            ("Victor", {"PER": "X", "LOC": "", "ORG": "", "MISC": ""}),
            ("Hugo",   {"PER": "X", "LOC": "", "ORG": "", "MISC": ""}),
            ("est",    {"PER": "",  "LOC": "", "ORG": "", "MISC": ""})
        ]
    """
    with open(chemin_csv, encoding='utf-8') as f:
        lignes = f.readlines()

    # La première ligne contient les noms des colonnes
    ligne_entete = lignes[0].strip()
    colonnes_entete = re.split(";", ligne_entete)
    # colonnes_entete = ["token", "PER", "LOC", "ORG", "MISC"]

    annotations = []
    for ligne in lignes[1:]:
        elements = re.split(";", ligne.strip())

        token = elements[0]  # La première colonne = le mot

        # Associer chaque étiquette à sa valeur pour ce token
        # zip aligne les deux listes : entete[1:] ↔ elements[1:]
        dic_token = {}
        for etiquette, valeur in zip(colonnes_entete[1:], elements[1:]):
            dic_token[etiquette] = valeur

        annotations.append((token, dic_token))

    return annotations


# ─────────────────────────────────────────────
# 2. CALCUL DES MÉTRIQUES
# ─────────────────────────────────────────────

def calculer_pr_re_f1(vp: int, fp: int, fn: int) -> tuple:
    """
    Calcule la Précision, le Rappel et le F1-score à partir
    des comptages VP, FP, FN.

    Paramètres :
        vp (int) : Vrais Positifs (correctement détectés)
        fp (int) : Faux Positifs (détectés à tort)
        fn (int) : Faux Négatifs (manqués)

    Retour :
        tuple : (précision, rappel, f1) — valeurs entre 0 et 1

    Gestion des cas limites :
        - Si VP+FP = 0 → Précision = 0 (évite la division par zéro)
        - Si VP+FN = 0 → Rappel = 0
        - Si VP = 0    → F1 = 0

    Exemples :
        >>> calculer_pr_re_f1(7, 3, 3)
        (0.7, 0.7, 0.7)

        >>> calculer_pr_re_f1(0, 5, 10)
        (0, 0, 0)   # Système qui ne trouve rien de correct
    """
    # Précision : parmi ce que le système a détecté, combien sont corrects ?
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0

    # Rappel : parmi ce qui existe réellement, combien a-t-il trouvé ?
    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0

    # F1 : moyenne harmonique de la précision et du rappel
    # La moyenne harmonique pénalise les déséquilibres entre P et R
    f1 = 0
    if vp > 0:
        f1 = 2 * precision * rappel / (precision + rappel)

    return precision, rappel, f1


# ─────────────────────────────────────────────
# 3. COMPARAISON GOLD vs SYSTÈME
# ─────────────────────────────────────────────

def comparer_csv(gold_csv: str, system_csv: str) -> tuple:
    """
    Compare les annotations humaines (gold) avec les prédictions
    du système (system) et calcule les métriques globales ET
    par catégorie d'entité (PER, LOC, ORG, MISC).

    Paramètres :
        gold_csv (str)   : chemin vers le CSV des annotations humaines
        system_csv (str) : chemin vers le CSV des prédictions système

    Retour :
        tuple : (dic_global, dic_par_categorie)
            - dic_global : {"VP": int, "FP": int, "FN": int}
            - dic_cat    : {"PER": {...}, "LOC": {...}, "ORG": {...}, "MISC": {...}}

    Algorithme token par token :
        Pour chaque token, on vérifie :
        1. Est-ce une entité dans le gold ? (gold_ent = True/False)
        2. Est-ce une entité dans le système ? (syst_ent = True/False)
        3. Pour chaque catégorie, VP/FP/FN selon le cas
    """
    gold = lire_annotations_csv(gold_csv)
    system = lire_annotations_csv(system_csv)

    # Vérification que les deux fichiers ont le même nombre de tokens
    if len(gold) != len(system):
        raise ValueError(
            f"Fichiers de tailles différentes : "
            f"gold={len(gold)}, system={len(system)}"
        )

    # Compteurs globaux
    dic_global = {"VP": 0, "FP": 0, "FN": 0}

    # Compteurs par catégorie
    categories = ["PER", "LOC", "ORG", "MISC"]
    dic_cat = {cat: {"VP": 0, "FP": 0, "FN": 0} for cat in categories}

    # Parcourir les tokens en parallèle (gold et system alignés)
    for (tok_g, annot_g), (tok_s, annot_s) in zip(gold, system):

        # ── Niveau GLOBAL (entité ou non-entité) ──
        # Est-ce une entité dans le gold ? (au moins une catégorie annotée)
        gold_ent = any(val != "" for val in annot_g.values())
        # Est-ce une entité dans le système ?
        syst_ent = any(val != "" for val in annot_s.values())

        if gold_ent and syst_ent:
            dic_global["VP"] += 1    # Trouvé correctement
        elif not gold_ent and syst_ent:
            dic_global["FP"] += 1    # Inventé à tort
        elif gold_ent and not syst_ent:
            dic_global["FN"] += 1    # Raté

        # ── Niveau CATÉGORIE ──
        # Identifier les catégories présentes dans chaque annotation
        gold_cats = {cat: (annot_g.get(cat, "") != "") for cat in categories}
        syst_cats = {cat: (annot_s.get(cat, "") != "") for cat in categories}

        for cat in categories:
            if gold_cats[cat] and syst_cats[cat]:
                dic_cat[cat]["VP"] += 1   # Bonne catégorie
            elif not gold_cats[cat] and syst_cats[cat]:
                dic_cat[cat]["FP"] += 1   # Mauvaise catégorie détectée
            elif gold_cats[cat] and not syst_cats[cat]:
                dic_cat[cat]["FN"] += 1   # Catégorie manquée

    return dic_global, dic_cat


def afficher_resultats(dic_global: dict, dic_cat: dict) -> None:
    """
    Affiche les métriques globales et par catégorie de façon lisible.

    Paramètres :
        dic_global (dict) : compteurs globaux {VP, FP, FN}
        dic_cat (dict)    : compteurs par catégorie
    """
    print("=" * 45)
    print("RÉSULTATS D'ÉVALUATION NER")
    print("=" * 45)

    # ── Global ──
    vp, fp, fn = dic_global["VP"], dic_global["FP"], dic_global["FN"]
    p, r, f1 = calculer_pr_re_f1(vp, fp, fn)
    print(f"\n📊 GLOBAL")
    print(f"  VP={vp:4d}  FP={fp:4d}  FN={fn:4d}")
    print(f"  Précision : {p:.4f} ({p*100:.1f}%)")
    print(f"  Rappel    : {r:.4f} ({r*100:.1f}%)")
    print(f"  F1-score  : {f1:.4f} ({f1*100:.1f}%)")

    # ── Par catégorie ──
    print(f"\n📋 PAR CATÉGORIE")
    print(f"  {'Cat':4}  {'P':6}  {'R':6}  {'F1':6}  VP    FP    FN")
    print(f"  {'-'*50}")
    for cat, valeurs in dic_cat.items():
        vp, fp, fn = valeurs["VP"], valeurs["FP"], valeurs["FN"]
        p, r, f1 = calculer_pr_re_f1(vp, fp, fn)
        print(f"  {cat:4}  {p:.4f}  {r:.4f}  {f1:.4f}  {vp:4d}  {fp:4d}  {fn:4d}")


# ─────────────────────────────────────────────
# 4. VISUALISATION AVEC MATPLOTLIB
# ─────────────────────────────────────────────

def tracer_metriques(dic_global: dict, dic_cat: dict) -> None:
    """
    Trace des graphiques en barres pour les métriques d'évaluation :
        - Graphique global (Précision, Rappel, F1)
        - Graphique par catégorie

    Paramètres :
        dic_global (dict) : compteurs globaux
        dic_cat (dict)    : compteurs par catégorie
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Graphique 1 : Métriques globales ──
    categories_axe = ["Précision", "Rappel", "F1-score"]
    vp, fp, fn = dic_global["VP"], dic_global["FP"], dic_global["FN"]
    p, r, f1 = calculer_pr_re_f1(vp, fp, fn)
    valeurs_globales = [p, r, f1]

    axes[0].bar(categories_axe, valeurs_globales,
                color=["steelblue", "orange", "green"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Métriques globales")
    axes[0].set_ylabel("Score (0 à 1)")
    # Afficher les valeurs au-dessus des barres
    for i, v in enumerate(valeurs_globales):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    # ── Graphique 2 : Métriques par catégorie ──
    cats = list(dic_cat.keys())
    precisions = []
    rappels = []
    f1s = []

    for cat in cats:
        vp = dic_cat[cat]["VP"]
        fp = dic_cat[cat]["FP"]
        fn = dic_cat[cat]["FN"]
        p, r, f1 = calculer_pr_re_f1(vp, fp, fn)
        precisions.append(p)
        rappels.append(r)
        f1s.append(f1)

    x = range(len(cats))
    width = 0.25  # Largeur des barres

    # 3 groupes de barres côte à côte pour chaque catégorie
    axes[1].bar([i - width for i in x], precisions, width, label="Précision", color="steelblue")
    axes[1].bar([i         for i in x], rappels,    width, label="Rappel",    color="orange")
    axes[1].bar([i + width for i in x], f1s,        width, label="F1",        color="green")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Métriques par catégorie")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("evaluation_ner.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("Graphique sauvegardé → evaluation_ner.png")


# ─────────────────────────────────────────────
# 5. EXEMPLE AVEC DES DONNÉES SIMULÉES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Test des métriques ──
    print("=== Test de calculer_pr_re_f1 ===")
    cas_tests = [
        (23, 5, 12,  "Bon système"),
        (23, 55, 12, "Beaucoup de FP"),
        (0, 5, 12,   "Aucun vrai positif"),
        (100, 10, 5, "Excellent système"),
    ]
    for vp, fp, fn, description in cas_tests:
        p, r, f1 = calculer_pr_re_f1(vp, fp, fn)
        print(f"\n  {description} (VP={vp}, FP={fp}, FN={fn})")
        print(f"  Précision={p:.3f}  Rappel={r:.3f}  F1={f1:.3f}")

    # ── Utilisation réelle ──
    print("\n=== Utilisation sur de vrais fichiers CSV ===")
    print("  dic_global, dic_cat = comparer_csv('gold.csv', 'system.csv')")
    print("  afficher_resultats(dic_global, dic_cat)")
    print("  tracer_metriques(dic_global, dic_cat)")
