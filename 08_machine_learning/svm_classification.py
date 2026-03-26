"""
=============================================================
08a — Classification avec SVM (Support Vector Machine)
=============================================================

CONCEPT CLÉ — SVM :
    Le SVM cherche le meilleur "hyperplan séparateur" entre
    les classes, en maximisant la MARGE (distance entre l'hyperplan
    et les points les plus proches de chaque classe).

    INTUITION GÉOMÉTRIQUE :
        ┌─────────────────────────────┐
        │  ○ ○   ┊     × ×           │
        │    ○   ┊   ×   ×           │
        │  ○     ┊     ×             │
        │        ┊                   │
        │    marge maximisée         │
        └─────────────────────────────┘

    AVANTAGES DU SVM :
        ✓ Fonctionne bien avec peu de données
        ✓ Robuste aux données de haute dimension
        ✓ Polyvalent : texte, images, numérique
        ✓ Avec le "kernel trick" → données non-linéaires

GÉNÉRALISATION :
    Toujours le même workflow sklearn :
        1. Préparer X (features) et y (labels)
        2. model = SVC()   ← créer le modèle
        3. model.fit(X, y) ← entraîner
        4. model.predict(X_nouveau) ← prédire

APPLICATIONS EN NLP :
    - Classificateur de sentiment (positif/négatif)
    - Détection de spam
    - Identification de registre (formel/informel)
    - Classification de questions
=============================================================
"""

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────
# 1. SVM SUR DONNÉES NUMÉRIQUES SIMPLES
# ─────────────────────────────────────────────

def exercice1_svm_numerique():
    """
    Exercice 1 : SVM basique sur des features numériques.

    DONNÉES :
        Chaque exemple = [longueur_phrase, nb_ponctuation]
        Classes : "neutre" ou "expressif"

    On observe que les phrases expressives ont tendance à
    être plus longues et à utiliser plus de ponctuation.
    """
    print("=" * 45)
    print("EXERCICE 1 — SVM sur données numériques")
    print("=" * 45)

    # Features : [longueur_phrase, nb_caracteres_speciaux]
    X = [
        [5,  0],   # Phrase courte, neutre
        [6,  0],   # Phrase courte, neutre
        [7,  1],   # Phrase médium, peu expressif
        [15, 3],   # Phrase longue, expressif
        [18, 4],   # Phrase longue, très expressif
        [20, 5],   # Phrase très longue, très expressif
    ]
    Y = ["neutre", "neutre", "neutre", "expressif", "expressif", "expressif"]

    # Créer et entraîner le SVM
    # SVC = Support Vector Classifier (version classification du SVM)
    modele = SVC()
    modele.fit(X, Y)  # Entraînement : le modèle apprend la frontière

    # Prédire sur de nouveaux exemples
    nouveaux = [[8, 0], [17, 4]]
    predictions = modele.predict(nouveaux)

    print(f"\nNouveau [8, 0] (phrase courte)   → Prédit : {predictions[0]}")
    print(f"Nouveau [17, 4] (phrase longue) → Prédit : {predictions[1]}")

    print("\nRéflexion :")
    print("  → [8, 0] : court et sans ponctuation → probablement neutre")
    print("  → [17, 4] : long avec ponctuation → probablement expressif")


# ─────────────────────────────────────────────
# 2. CLASSIFICATION QUESTION / AFFIRMATION
# ─────────────────────────────────────────────

def exercice2_question_affirmation():
    """
    Exercice 2 : Classifier des phrases comme questions ou affirmations
    à partir de deux indices linguistiques booléens.

    FEATURES :
        x[0] = présence d'un point d'interrogation (0 ou 1)
        x[1] = présence d'un mot interrogatif : pourquoi, comment, etc. (0 ou 1)

    OBSERVATIONS INTÉRESSANTES :
        [1, 0] → point d'interrogation SANS mot interrogatif
                 Ex : "Tu viens ?" → quand même une question !
        [0, 1] → mot interrogatif SANS point d'interrogation
                 Ambigu... Peut être une affirmation indirecte.
    """
    print("\n" + "=" * 45)
    print("EXERCICE 2 — Question ou affirmation ?")
    print("=" * 45)

    # [présence_?, présence_mot_interrogatif]
    X = [
        [1, 1],  # "Pourquoi viens-tu ?"       → question
        [1, 0],  # "Tu viens ?"                 → question
        [0, 0],  # "Tu viens."                  → affirmation
        [0, 0],  # "Je suis là."                → affirmation
        [1, 1],  # "Comment ça va ?"            → question
        [0, 0],  # "Il fait beau."              → affirmation
    ]
    y = ["question", "question", "affirmation", "affirmation", "question", "affirmation"]

    modele = SVC()
    modele.fit(X, y)

    # Cas de test avec analyse des résultats
    tests = [[1, 0], [0, 1], [0, 0]]
    labels_tests = ["[1,0] point d'interrogation seul",
                    "[0,1] mot interrogatif seul (ambigu !)",
                    "[0,0] ni point ni mot interrogatif"]

    print("\nPrédictions :")
    for test, label, pred in zip(tests, labels_tests, modele.predict(tests)):
        print(f"  {label} → {pred}")

    print("\nAnalyse :")
    print("  → [0, 1] est le cas le plus ambigu :")
    print("    'Il demande comment ça va.' = affirmation avec mot interrogatif")
    print("    'Comment ça va ?' = question sans point d'interrogation")


# ─────────────────────────────────────────────
# 3. CLASSIFICATION REGISTRE FORMEL/INFORMEL
# ─────────────────────────────────────────────

def exercice3_registre():
    """
    Exercice 3 : Détecter le registre (formel vs informel).

    FEATURES (3 dimensions) :
        x[0] = présence d'abréviations (slt, bjr, cc...) : 0 ou 1
        x[1] = présence de "vous" : 0 ou 1
        x[2] = présence d'émoticônes : 0 ou 1

    OBSERVATION :
        Les indices peuvent se mélanger → contexte hybride possible.
        Ex : "Bonjour :)" → formel (bonjour) + informel (:))
    """
    print("\n" + "=" * 45)
    print("EXERCICE 3 — Registre formel / informel")
    print("=" * 45)

    # [abréviations, vous, émoticônes]
    X = [
        [1, 0, 1],  # "slt :)"          → informel
        [1, 0, 0],  # "bjr"             → informel
        [0, 1, 0],  # "Pouvez-vous ?"   → formel
        [0, 1, 0],  # "Je vous remercie"→ formel
        [1, 0, 1],  # "cc :)"           → informel
        [0, 1, 0],  # "Veuillez..."     → formel
    ]
    y = ["informel", "informel", "formel", "formel", "informel", "formel"]

    modele = SVC()
    modele.fit(X, y)

    # Tests avec analyse
    tests = [[1, 0, 0], [0, 1, 1]]
    descriptions = [
        "[1,0,0] abréviation seule (sans émoticône)",
        "[0,1,1] 'vous' + émoticône (registre mixte !)"
    ]

    print("\nPrédictions :")
    predictions = modele.predict(tests)
    for desc, pred in zip(descriptions, predictions):
        print(f"  {desc} → {pred}")

    print("\nAnalyse :")
    print("  → [0,1,1] 'vous' + émoticône : le modèle doit trancher")
    print("    En pratique, ce cas montre les limites d'une représentation binaire simple")


# ─────────────────────────────────────────────
# 4. ANALYSE DE SENTIMENT (POSITIF / NÉGATIF)
# ─────────────────────────────────────────────

def exercice4_sentiment():
    """
    Exercice 4 : Classification de sentiment basée sur le
    comptage de mots positifs et négatifs.

    FEATURES :
        x[0] = nombre de mots positifs (bien, super, excellent...)
        x[1] = nombre de mots négatifs (mauvais, nul, horrible...)

    OBSERVATION SUR L'AMBIGUITÉ :
        [1, 1] = 1 mot positif, 1 mot négatif → très ambigu !
        Le SVM tracera une frontière, mais le résultat peut varier.
    """
    print("\n" + "=" * 45)
    print("EXERCICE 4 — Sentiment positif / négatif")
    print("=" * 45)

    # [nb_mots_positifs, nb_mots_negatifs]
    X = [
        [3, 0],   # Très positif
        [2, 0],   # Positif
        [0, 3],   # Très négatif
        [0, 2],   # Négatif
        [2, 1],   # Plutôt positif
        [1, 2],   # Plutôt négatif
    ]
    y = ["positif", "positif", "negatif", "negatif", "positif", "negatif"]

    modele = SVC()
    modele.fit(X, y)

    tests = [[3, 1], [0, 4], [1, 1]]
    descriptions = [
        "[3, 1] 3 positifs, 1 négatif → probablement positif",
        "[0, 4] seulement négatifs → clairement négatif",
        "[1, 1] équilibre parfait → AMBIGU (dépend du modèle)"
    ]

    print("\nPrédictions :")
    predictions = modele.predict(tests)
    for desc, pred in zip(descriptions, predictions):
        print(f"  {desc}")
        print(f"  → Prédit : {pred}\n")


# ─────────────────────────────────────────────
# 5. SVM AVEC TEXTE BRUT (CountVectorizer)
# ─────────────────────────────────────────────

def exercice5_svm_texte():
    """
    Exercice 5 : Pipeline complet texte → vecteur → SVM.

    CountVectorizer transforme automatiquement des textes en
    vecteurs de fréquences de mots (bag of words).

    PIPELINE :
        Textes bruts
            ↓ CountVectorizer.fit_transform()
        Matrice creuse (sparse) : chaque ligne = un texte,
                                  chaque colonne = un mot du vocabulaire,
                                  chaque valeur = fréquence du mot
            ↓ SVC.fit()
        Modèle entraîné
            ↓ predict()
        Prédictions

    IMPORTANT : On utilise .transform() (pas .fit_transform())
    sur les nouveaux textes pour garder le même vocabulaire.
    """
    print("\n" + "=" * 45)
    print("EXERCICE 5 — SVM sur texte brut avec CountVectorizer")
    print("=" * 45)

    # Corpus d'entraînement
    textes_train = [
        "bonjour madame je vous remercie",   # formel
        "salut ça va",                        # informel
        "veuillez recevoir mes salutations",  # formel
        "coucou merci :)",                    # informel
        "je vous prie d accepter",            # formel
        "hello :)"                            # informel
    ]
    labels = ["formel", "informel", "formel", "informel", "formel", "informel"]

    # Étape 1 : Transformer les textes en vecteurs numériques
    # CountVectorizer crée un vocabulaire et compte les occurrences
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(textes_train)
    # fit_transform = apprendre le vocabulaire ET transformer

    print(f"\nVocabulaire appris ({len(vectorizer.vocabulary_)} mots) :")
    print(f"  {sorted(vectorizer.vocabulary_.keys())[:10]}...")

    # Étape 2 : Entraîner le SVM sur les vecteurs
    modele = SVC()
    modele.fit(X_train, labels)

    # Étape 3 : Prédire sur de nouveaux textes
    nouveaux_textes = ["bonjour", "coucou ça va", "je vous remercie"]
    # IMPORTANT : transform (sans fit) → même vocabulaire que l'entraînement
    X_test = vectorizer.transform(nouveaux_textes)

    predictions = modele.predict(X_test)

    print("\nPrédictions :")
    for texte, pred in zip(nouveaux_textes, predictions):
        print(f"  '{texte}' → {pred}")

    print("\nExplication :")
    print("  'bonjour' → formel (présent dans l'entraînement formel)")
    print("  'coucou ça va' → informel (mots informels)")
    print("  'je vous remercie' → formel (mots formels connus)")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    exercice1_svm_numerique()
    exercice2_question_affirmation()
    exercice3_registre()
    exercice4_sentiment()
    exercice5_svm_texte()

    print("\n" + "=" * 45)
    print("RÉSUMÉ DU WORKFLOW SVM")
    print("=" * 45)
    print("""
    from sklearn.svm import SVC
    from sklearn.feature_extraction.text import CountVectorizer

    # Sur données numériques :
    modele = SVC()
    modele.fit(X_train, y_train)
    predictions = modele.predict(X_test)

    # Sur texte brut :
    vect = CountVectorizer()
    X_train = vect.fit_transform(textes_train)
    X_test  = vect.transform(textes_test)    # ← transform seulement !
    modele = SVC()
    modele.fit(X_train, y_train)
    modele.predict(X_test)
    """)
