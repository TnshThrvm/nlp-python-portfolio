# 🧠 NLP & IA — Portfolio Python

> Implémentations commentées de concepts fondamentaux en Traitement Automatique des Langues (TAL/NLP) et Intelligence Artificielle.  
> Réalisé dans le cadre de cursus universitaire à Sorbonne Université.

---

## 📁 Structure du projet

```
nlp-python-portfolio/
│
├── 01_text_processing/          # Lecture et nettoyage de fichiers texte
├── 02_zipf_statistics/          # Statistiques textuelles et loi de Zipf
├── 03_language_identification/  # Identification de langue par n-grammes
├── 04_named_entity_recognition/ # Reconnaissance d'entités nommées (SpaCy)
├── 05_evaluation_metrics/       # Précision, Rappel, F1 — évaluation NER
├── 06_language_models/          # Modèles unigramme et bigramme
├── 07_word_embeddings/          # Word2Vec, GloVe, similarité cosinus
├── 08_machine_learning/
│   ├── svm/                     # Support Vector Machines
│   ├── kmeans/                  # Clustering K-Means
│   └── neural_networks/         # Réseaux de neurones (MLP)
├── 09_logic_Z3/                 # Logique du premier ordre avec Z3
└── docs/                        # Guides et fiches récapitulatives
```

---

## 🚀 Installation rapide

```bash
# Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/nlp-python-portfolio.git
cd nlp-python-portfolio

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les modèles SpaCy
python -m spacy download fr_core_news_sm
python -m spacy download fr_core_news_lg
```

---

## 📦 Dépendances

| Bibliothèque | Usage |
|---|---|
| `spacy` | Reconnaissance d'entités nommées |
| `scikit-learn` | SVM, K-Means, MLP, métriques |
| `gensim` | Word embeddings (Word2Vec, GloVe) |
| `sentence-transformers` | Embeddings de phrases |
| `z3-solver` | Logique du premier ordre |
| `matplotlib` | Visualisations |
| `plotly` | Graphiques interactifs |
| `pandas` | Manipulation de données |
| `numpy` | Calcul numérique |
| `beautifulsoup4` | Parsing HTML |

---

## 🗺️ Parcours recommandé

**Débutant** → `01` → `02` → `03`  
**Intermédiaire** → `04` → `05` → `06` → `07`  
**Avancé** → `08` → `09`

---

## 📚 Concepts couverts

- **Linguistique computationnelle** : loi de Zipf, n-grammes, modèles de langue
- **NER** : extraction d'entités avec SpaCy, évaluation VP/FP/FN
- **Représentations vectorielles** : TF-IDF, Word2Vec, GloVe, sentence embeddings
- **Classification** : SVM, réseaux de neurones (MLP)
- **Clustering** : K-Means sur texte et embeddings
- **Logique formelle** : FOL avec Z3, base de connaissances, inférence

---

*Sorbonne Université — UFR de Linguistique*
