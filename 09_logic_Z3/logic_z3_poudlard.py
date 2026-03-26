"""
=============================================================
09 — Logique du premier ordre avec Z3
=============================================================

CONCEPT CLÉ — LOGIQUE DU PREMIER ORDRE (FOL) :
    Langage formel pour représenter des connaissances et raisonner
    de façon automatique.

    COMPOSANTS :
        - Sorts (types) : Personne, Cours, Maison
        - Constantes    : harry, ron, potions
        - Prédicats     : Eleve(x), Prof(x), Ami(x,y)
        - Fonctions     : meilleurAmi(x) → Personne
        - Quantificateurs : ∀x, ∃x

    RAISONNEMENT :
        KB ⊨ φ signifie que φ est vrai dans TOUT modèle de KB
        Pour vérifier : tester si KB ∧ ¬φ est insatisfiable (UNSAT)

Z3 (Microsoft Research) :
    Solveur SMT (Satisfiability Modulo Theories)
    → Vérifie automatiquement si une KB est satisfiable
    → Trouve des modèles concrets
    → Prouve des conséquences logiques

INSTALLATION :
    pip install z3-solver

HYPOTHÈSES IMPORTANTES :
    UNA (Unique Name Assumption) : des constantes différentes
        désignent des individus différents → Distinct(harry, ron)
    DCA (Domain Closure Assumption) : le domaine est fermé
        → ForAll(x, Or(x == harry, x == ron))
    Complétion de prédicat : si seul harry est préfet →
        ForAll(x, Implies(Prefet(x), x == harry))
=============================================================
"""

from z3 import *


# ─────────────────────────────────────────────
# PARTIE 1 : VOCABULAIRE DE POUDLARD
# ─────────────────────────────────────────────

def construire_vocabulaire_poudlard():
    """
    Déclare le vocabulaire (signature) du monde de Poudlard.

    SORTS : types des objets
        Eleve, Professeur, Cours, Maison
        (chaque sort = un domaine d'objets séparé)

    CONSTANTES : individus spécifiques
        Harry, Ron, etc. → des objets du sort Eleve
        Rogue → un objet du sort Professeur

    PRÉDICATS : propriétés et relations
        Prefet(x)           : x est préfet → Bool
        AppartientA(x, m)   : x appartient à la maison m → Bool
        Enseigne(p, c)      : p enseigne le cours c → Bool

    FONCTIONS : mappings entre objets
        meilleurAmi(x) → Personne : le meilleur ami de x

    Retour :
        dict : tous les symboles du vocabulaire
    """
    # ── Sorts ──
    # DeclareSort crée un nouveau type "abstrait"
    Eleve_sort      = DeclareSort('Eleve')
    Professeur_sort = DeclareSort('Prof')
    Cours_sort      = DeclareSort('Cours')
    Maison_sort     = DeclareSort('Maison')

    # ── Constantes ──
    # Const('nom', Sort) = un objet du sort donné
    harry   = Const('harry',   Eleve_sort)
    ron     = Const('ron',     Eleve_sort)
    drago   = Const('drago',   Eleve_sort)
    rogue   = Const('rogue',   Professeur_sort)
    mcgo    = Const('mcgonagall', Professeur_sort)
    potions = Const('potions', Cours_sort)
    metamor = Const('metamorphose', Cours_sort)
    gryffo  = Const('gryffondor', Maison_sort)
    serpen  = Const('serpentard', Maison_sort)

    # ── Prédicats (fonctions → Bool) ──
    Prefet      = Function('Prefet',      Eleve_sort,      BoolSort())
    AppartientA = Function('AppartientA', Eleve_sort, Maison_sort, BoolSort())
    DirecteurDe = Function('DirecteurDe', Professeur_sort, Maison_sort, BoolSort())
    Enseigne    = Function('Enseigne',    Professeur_sort, Cours_sort, BoolSort())

    print("✓ Vocabulaire Poudlard créé")
    print(f"  Sorts      : Eleve, Prof, Cours, Maison")
    print(f"  Constantes : harry, ron, drago, rogue, mcgonagall, potions, metamorphose, gryffondor, serpentard")
    print(f"  Prédicats  : Prefet, AppartientA, DirecteurDe, Enseigne")

    return {
        'sorts': (Eleve_sort, Professeur_sort, Cours_sort, Maison_sort),
        'constantes': (harry, ron, drago, rogue, mcgo, potions, metamor, gryffo, serpen),
        'predicats': (Prefet, AppartientA, DirecteurDe, Enseigne)
    }


# ─────────────────────────────────────────────
# PARTIE 2 : BASE DE CONNAISSANCES POUDLARD
# ─────────────────────────────────────────────

def construire_kb_poudlard():
    """
    Construit une base de connaissances complète sur Poudlard.

    FAITS DE BASE (assertions atomiques) :
        Prefet(harry), Prefet(ron), Enseigne(rogue, potions)...

    CONNECTEURS LOGIQUES Z3 :
        And(φ, ψ)        → φ ∧ ψ  (conjonction)
        Or(φ, ψ)         → φ ∨ ψ  (disjonction)
        Not(φ)           → ¬φ     (négation)
        Implies(φ, ψ)    → φ → ψ  (implication)
        φ == ψ           → φ ↔ ψ  (équivalence / égalité)

    QUANTIFICATEURS Z3 :
        ForAll([x], φ(x)) → ∀x. φ(x)
        Exists([x], φ(x)) → ∃x. φ(x)
        ⚠️ La variable x doit être déclarée avec Const()

    Retour :
        tuple : (KB, vocabulaire)
    """
    # ── Vocabulaire partagé (sort unique 'Personne' pour simplifier) ──
    Personne = DeclareSort('Personne')
    Cours    = DeclareSort('Cours')
    Maison   = DeclareSort('Maison')

    # Constantes
    harry, ron, hermione, drago, rogue = Consts(
        'harry ron hermione drago rogue', Personne
    )
    potions, metamorphose = Consts('potions metamorphose', Cours)
    gryffondor, serpentard = Consts('gryffondor serpentard', Maison)

    # Variables pour les quantificateurs
    x = Const('x', Personne)
    y = Const('y', Personne)

    # Fonctions et prédicats
    meilleurAmi = Function('meilleurAmi', Personne, Personne)
    estDirigePar = Function('estDirigePar', Maison, Personne)
    Eleve    = Function('Eleve',    Personne, BoolSort())
    Prof     = Function('Prof',     Personne, BoolSort())
    Ami      = Function('Ami',      Personne, Personne, BoolSort())
    Enseigne = Function('Enseigne', Personne, Cours, BoolSort())
    AReussi  = Function('AReussi',  Personne, Cours, BoolSort())

    # ── Créer la KB ──
    KB = Solver()

    # Faits atomiques
    KB.add(Eleve(harry))
    KB.add(Eleve(ron))
    KB.add(Eleve(hermione))
    KB.add(Prof(rogue))

    # Égalité et inégalité
    KB.add(meilleurAmi(ron) == harry)        # Le meilleur ami de Ron est Harry
    KB.add(meilleurAmi(harry) != drago)      # Harry n'est pas l'ami de Drago

    # Relations
    KB.add(Enseigne(rogue, potions))         # Rogue enseigne Potions
    KB.add(AReussi(harry, potions))          # Harry a réussi Potions
    KB.add(Not(AReussi(ron, potions)))       # Ron n'a PAS réussi Potions

    # Connecteurs logiques
    KB.add(Ami(ron, harry))                          # Ron est ami de Harry
    KB.add(And(Ami(ron, harry), Ami(harry, hermione)))  # Et Harry est ami de Hermione
    KB.add(Not(Ami(harry, drago)))                   # Harry et Drago ne sont PAS amis
    KB.add(Or(AReussi(ron, potions),
              AReussi(ron, metamorphose)))            # Ron a réussi au moins un cours

    # Implication
    KB.add(Implies(AReussi(harry, potions),
                   AReussi(harry, metamorphose)))    # Harry réussit Potions → réussit Métamorphose

    # Quantificateurs
    # ∀x : Prof(x) → ¬Eleve(x)  (personne ne peut être les deux)
    KB.add(ForAll(x, Implies(Prof(x), Not(Eleve(x)))))

    # ∀x : Eleve(x) → AReussi(x, potions) sauf Harry
    KB.add(ForAll(x, Implies(And(Eleve(x), x != harry),
                              AReussi(x, potions))))

    # La relation Ami est symétrique : Ami(x,y) ↔ Ami(y,x)
    KB.add(ForAll([x, y], Implies(Ami(x, y), Ami(y, x))))

    print("✓ KB Poudlard construite")
    print(f"  Formules ajoutées : {len(KB.assertions())}")

    return KB, {
        'personnes': (harry, ron, hermione, drago, rogue),
        'cours': (potions, metamorphose),
        'predicats': (Eleve, Prof, Ami, Enseigne, AReussi),
        'variables': (x, y)
    }


# ─────────────────────────────────────────────
# PARTIE 3 : RAISONNEMENT ET INFÉRENCE
# ─────────────────────────────────────────────

def raisonner_avec_kb():
    """
    Démontre le raisonnement automatique avec Z3.

    MÉTHODE : Pour prouver KB ⊨ φ
        1. KB.push()           → sauvegarder l'état
        2. KB.add(Not(φ))      → ajouter la négation de φ
        3. KB.check()          → si UNSAT → φ est prouvée !
        4. KB.pop()            → restaurer l'état original

    POURQUOI NOT(φ) ?
        Si KB ∧ ¬φ est insatisfiable (aucun modèle)
        → il est impossible que KB soit vraie et φ fausse
        → donc KB implique forcément φ
    """
    Personne = DeclareSort('Personne')
    Cours    = DeclareSort('Cours')

    harry, ron = Consts('harry ron', Personne)
    potions, metamorphose = Consts('potions metamorphose', Cours)

    Eleve   = Function('Eleve',   Personne, BoolSort())
    AReussi = Function('AReussi', Personne, Cours, BoolSort())

    # Base de connaissances de raisonnement
    KB2 = Solver()
    KB2.add(Eleve(harry))
    KB2.add(Eleve(ron))
    KB2.add(AReussi(harry, potions))
    KB2.add(Not(AReussi(ron, potions)))
    # Règle : si Harry réussit Potions, il réussit Métamorphose
    KB2.add(Implies(AReussi(harry, potions),
                    AReussi(harry, metamorphose)))

    print("=== RAISONNEMENT AVEC Z3 ===\n")

    # ── Vérifier la satisfiabilité ──
    resultat = KB2.check()
    print(f"La KB est : {resultat}")
    # → sat : il existe une interprétation qui satisfait la KB

    if resultat == sat:
        print("\nUn modèle possible :")
        print(KB2.model())

    # ── Test 1 : Harry a-t-il réussi Métamorphose ? ──
    print("\n--- Test : KB ⊨ AReussi(harry, metamorphose) ? ---")
    phi = AReussi(harry, metamorphose)

    KB2.push()                     # Sauvegarder l'état
    KB2.add(Not(phi))              # Tester la négation
    if KB2.check() == unsat:
        print(f"✓ OUI ! AReussi(harry, metamorphose) est une conséquence logique")
        print(f"  (La négation est UNSAT → l'affirmation est nécessairement vraie)")
    else:
        print(f"✗ NON, voici un contre-exemple :")
        print(KB2.model())
    KB2.pop()                      # Restaurer l'état

    # ── Test 2 : Ron a-t-il réussi Métamorphose ? ──
    print("\n--- Test : KB ⊨ AReussi(ron, metamorphose) ? ---")
    psi = AReussi(ron, metamorphose)

    KB2.push()
    KB2.add(Not(psi))
    resultat2 = KB2.check()
    if resultat2 == unsat:
        print(f"✓ OUI ! Conséquence logique.")
    else:
        print(f"✗ NON, contre-exemple :")
        print(KB2.model())
        print(f"  → La KB ne dit rien sur AReussi(ron, métamorphose)")
        print(f"  → Monde ouvert : c'est possible mais pas obligatoire")
    KB2.pop()


# ─────────────────────────────────────────────
# PARTIE 4 : HYPOTHÈSES SPÉCIALES (UNA, DCA, CP)
# ─────────────────────────────────────────────

def demonstrer_hypotheses():
    """
    Démontre les trois hypothèses qui ne sont PAS automatiques en FOL.

    1. UNA (Unique Name Assumption) :
       Sans UNA, harry et ron pourraient être le même objet !
       Solution : Distinct(harry, ron, ...)

    2. DCA (Domain Closure Assumption) :
       Sans DCA, Z3 peut inventer des individus supplémentaires.
       Solution : ForAll(x, Or(x == harry, x == ron))

    3. Complétion de prédicat :
       Sans complétion, Z3 peut rendre un prédicat vrai pour
       n'importe qui même si on ne l'a pas explicitement déclaré.
       Solution : ForAll(x, Implies(Prefet(x), x == harry))
    """
    Personne = DeclareSort('Personne')
    harry, ron = Consts('harry ron', Personne)
    x = Const('x', Personne)

    Eleve   = Function('Eleve',   Personne, BoolSort())
    Prefet  = Function('Prefet',  Personne, BoolSort())
    ReussiAnnee = Function('ReussiAnnee', Personne, BoolSort())

    print("\n=== HYPOTHÈSE 1 : UNA (Noms Uniques) ===\n")

    KB1 = Solver()
    KB1.add(Eleve(harry))
    KB1.add(Eleve(ron))
    KB1.add(ForAll(x, Implies(Eleve(x), And(x != harry,
                                             ReussiAnnee(x)))))

    print("SANS UNA :")
    KB1.check()
    print(f"  Modèle : {KB1.model()}")
    print(f"  ⚠️ harry et ron peuvent être le MÊME objet !")

    # Ajouter l'hypothèse UNA
    KB1.add(Distinct(harry, ron))
    print("\nAVEC UNA (Distinct(harry, ron)) :")
    KB1.check()
    print(f"  Modèle : {KB1.model()}")
    print(f"  ✓ harry et ron sont maintenant des objets distincts")

    print("\n=== HYPOTHÈSE 2 : DCA (Clôture du Domaine) ===\n")

    KB2 = Solver()
    KB2.add(Eleve(harry))
    KB2.add(Eleve(ron))
    # Formule : ce n'est PAS vrai que tout le monde est un élève
    KB2.add(Not(ForAll(x, Eleve(x))))

    print("SANS DCA :")
    KB2.check()
    print(f"  Modèle : {KB2.model()}")
    print(f"  ⚠️ Z3 peut inventer un 3e individu qui n'est pas élève")

    # Ajouter DCA : seuls harry et ron existent
    KB2.add(ForAll(x, Or(x == harry, x == ron)))
    print("\nAVEC DCA (ForAll(x, Or(x==harry, x==ron))) :")
    if KB2.check() == unsat:
        print(f"  ✓ UNSAT : si tout le monde est harry ou ron,")
        print(f"    et les deux sont élèves, ¬∀x.Eleve(x) est impossible")
    else:
        print(f"  Modèle : {KB2.model()}")

    print("\n=== HYPOTHÈSE 3 : Complétion de prédicat ===\n")

    KB3 = Solver()
    KB3.add(Eleve(harry))
    KB3.add(Eleve(ron))
    KB3.add(Prefet(harry))  # Seul Harry est préfet (mais Z3 ne le sait pas encore)

    print("SANS COMPLÉTION :")
    KB3.check()
    print(f"  Modèle : {KB3.model()}")
    print(f"  ⚠️ Ron peut aussi être Prefet (Z3 ne dit pas le contraire)")

    # Complétion : seul harry peut être préfet
    KB3.add(ForAll(x, Implies(Prefet(x), x == harry)))
    print("\nAVEC COMPLÉTION (ForAll(x, Prefet(x)→x==harry)) :")
    KB3.check()
    print(f"  Modèle : {KB3.model()}")
    print(f"  ✓ Prefet vaut True seulement pour harry")


# ─────────────────────────────────────────────
# PARTIE 5 : EXERCICE SANG-PUR (RAISONNEMENT COMPLEXE)
# ─────────────────────────────────────────────

def exercice_sang_pur():
    """
    Raisonnement complexe : Fleamont est-il un sorcier ?

    FAITS :
        1. Tout parent est un ancêtre
        2. La relation ancêtre est transitive
        3. Si quelqu'un est Sang-Pur, tous ses ancêtres sont des sorciers
        4. James est le père de Harry
        5. Lily est la mère de Harry
        6. Harry est un Sang-Pur
        7. Fleamont est le père de James

    QUESTION : Fleamont est-il nécessairement un sorcier ?
    RÉPONSE ATTENDUE : OUI
        → Fleamont est parent de James
        → Fleamont est donc ancêtre de James
        → James est parent de Harry
        → James est ancêtre de Harry
        → Par transitivité, Fleamont est ancêtre de Harry
        → Harry est Sang-Pur
        → Tous les ancêtres de Harry sont sorciers
        → Fleamont est sorcier ✓
    """
    print("\n=== EXERCICE : SANG-PUR ===\n")

    Personne = DeclareSort('Personne')

    harry, james, lily, fleamont = Consts(
        'harry james lily fleamont', Personne
    )
    x, y, z = Consts('x y z', Personne)

    Parent   = Function('Parent',   Personne, Personne, BoolSort())
    Ancetre  = Function('Ancetre',  Personne, Personne, BoolSort())
    SangPur  = Function('SangPur',  Personne, BoolSort())
    Sorcier  = Function('Sorcier',  Personne, BoolSort())

    KB_sangpur = Solver()

    # Fait 1 : Tout parent est un ancêtre
    KB_sangpur.add(ForAll([x, y],
        Implies(Parent(x, y), Ancetre(x, y))
    ))

    # Fait 2 : Transitivité des ancêtres
    KB_sangpur.add(ForAll([x, y, z],
        Implies(And(Ancetre(x, y), Ancetre(y, z)),
                Ancetre(x, z))
    ))

    # Fait 3 : Sang-Pur → tous ses ancêtres sont sorciers
    KB_sangpur.add(ForAll([x, y],
        Implies(And(SangPur(x), Ancetre(y, x)),
                Sorcier(y))
    ))

    # Faits 4-7 : Famille Potter
    KB_sangpur.add(Parent(james,   harry))    # James est parent de Harry
    KB_sangpur.add(Parent(lily,    harry))    # Lily est parent de Harry
    KB_sangpur.add(SangPur(harry))            # Harry est Sang-Pur
    KB_sangpur.add(Parent(fleamont, james))   # Fleamont est parent de James

    # Noms uniques (UNA)
    KB_sangpur.add(Distinct(harry, james, lily, fleamont))

    # ── Question : KB ⊨ Sorcier(fleamont) ? ──
    question = Sorcier(fleamont)
    KB_sangpur.push()
    KB_sangpur.add(Not(question))

    if KB_sangpur.check() == unsat:
        print("✓ OUI ! Fleamont est nécessairement un sorcier")
        print("  Chaîne de raisonnement :")
        print("    Fleamont → parent → James")
        print("    James    → parent → Harry")
        print("    Fleamont → ancêtre → James → ancêtre → Harry (transitivité)")
        print("    Harry    → Sang-Pur")
        print("    Donc tous les ancêtres de Harry → Sorciers")
        print("    Fleamont est ancêtre de Harry → Fleamont est Sorcier ✓")
    else:
        print("✗ NON, contre-exemple :")
        print(KB_sangpur.model())

    KB_sangpur.pop()


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("LOGIQUE DU PREMIER ORDRE AVEC Z3")
    print("=" * 55)

    try:
        # Partie 1 : Vocabulaire
        construire_vocabulaire_poudlard()

        # Partie 2 : KB + satisfiabilité
        print("\n" + "─" * 55)
        KB, vocab = construire_kb_poudlard()
        print(f"Satisfiabilité : {KB.check()}")

        # Partie 3 : Raisonnement
        print("\n" + "─" * 55)
        raisonner_avec_kb()

        # Partie 4 : Hypothèses UNA, DCA, CP
        print("\n" + "─" * 55)
        demonstrer_hypotheses()

        # Partie 5 : Exercice complexe
        print("\n" + "─" * 55)
        exercice_sang_pur()

    except Exception as e:
        print(f"Erreur : {e}")
        print("Assurez-vous d'avoir z3-solver installé : pip install z3-solver")

    print("\n" + "=" * 55)
    print("RÉSUMÉ : FORMULES Z3 ESSENTIELLES")
    print("=" * 55)
    print("""
    # Types
    S = DeclareSort('S')            # Déclarer un sort

    # Constantes et variables
    a = Const('a', S)               # Constante
    x = Const('x', S)               # Variable (même syntaxe)

    # Prédicats et fonctions
    P = Function('P', S, BoolSort())          # Prédicat unaire
    R = Function('R', S, S, BoolSort())       # Relation binaire
    f = Function('f', S, S)                   # Fonction

    # Connecteurs
    And(p, q)           # p ∧ q
    Or(p, q)            # p ∨ q
    Not(p)              # ¬p
    Implies(p, q)       # p → q
    p == q              # p ↔ q (ou égalité entre termes)

    # Quantificateurs
    ForAll([x], phi)    # ∀x. φ
    Exists([x], phi)    # ∃x. φ

    # Solveur
    KB = Solver()
    KB.add(formule)
    KB.check()           # sat / unsat / unknown
    KB.model()           # Un modèle satisfaisant
    KB.push() / KB.pop() # Sauvegarde/restauration d'état

    # Hypothèses
    Distinct(a, b, c)    # UNA : a, b, c sont distincts
    """)
