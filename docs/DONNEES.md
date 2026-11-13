# Données lourdes : où les mettre ?

GitHub n'est pas fait pour stocker de gros fichiers binaires : un fichier de plus de 100 Mo est refusé au `git push`, et un dépôt trop volumineux devient lent à cloner. Les données de ce projet sont donc exclues du dépôt (`.gitignore`) et doivent être hébergées ailleurs.

## Ce qui existe et sa taille approximative

| Dossier / fichier | Contenu | Taille approx. |
|---|---|---|
| `embeddings_v2/cache_semantique/cache_faiss.bin` | Index FAISS du cache Q/R | 287 Mo |
| `embeddings_v2/faiss_index.bin` | Index dense du corpus | 63 Mo |
| `embeddings_v2/chunk_map.json` | Texte et métadonnées des chunks | 62 Mo |
| `embeddings_v2/embeddings.npz` | Vecteurs des chunks | 58 Mo |
| `embeddings_v2/cache_semantique/cache_map.json` | Questions/réponses du cache | 41 Mo |
| `embeddings_v2/graph_index.json` | Graphe de métadonnées | 20 Mo |
| `embeddings_v2/bm25_index.pkl` | Index BM25 | 16 Mo |
| `chunks_output_v2/*.json` (46 fichiers) | Chunks par document | ≈ 100 Mo au total |
| `ragas_dataset_finetuned.jsonl` | Dataset d'évaluation RAGAS | 7,7 Mo |

## Options d'hébergement

| Option | Avantages | Limites |
|---|---|---|
| **Zenodo** | Donne un **DOI citable**, idéal pour un mémoire ; stockage pérenne | Dépôt manuel |
| **Hugging Face Hub** (dataset) | Pensé pour les gros fichiers ML, versionné, téléchargeable par script | Compte requis |
| **GitHub Releases** | Simple, reste associé au dépôt | Taille par fichier limitée ; pas de versionnement fin |
| **Git LFS** | Les fichiers restent dans le dépôt | Quota gratuit limité, à vérifier selon ton plan |

Recommandation : héberger les index sur **Zenodo ou Hugging Face**, et indiquer le lien dans le README. Les index (FAISS, BM25, graphe) sont **reconstructibles** à partir des chunks, et les chunks à partir des PDF, donc l'essentiel pour la reproductibilité est le code et la documentation du processus.

## Reconstruire les données

1. Placer les PDF dans `documents/` (ou définir `DOCUMENTS_DIR`).
2. `python src/chunking.py` → crée `chunks_output_v2/`.
3. Lancer le script d'indexation → crée `embeddings_v2/` (script à ajouter, voir README).
4. `python src/build_cache_semantique.py` → crée `embeddings_v2/cache_semantique/`.

## Avant de redistribuer

Les rapports sources sont des publications d'institutions statistiques. Vérifie leurs conditions de réutilisation avant de publier les PDF ou les chunks (qui contiennent leur texte). En cas de doute, publie seulement le code, les résultats d'évaluation et la liste des documents avec leur référence.
