# Processus détaillé du pipeline ANSTAT-IA

Ce document décrit, étape par étape, ce que fait chaque script, avec les paramètres réellement utilisés dans le code. Il sert aussi de support pour la partie méthodologie du mémoire.

Sommaire

1. [Principe d'un RAG](#1-principe-dun-rag)
2. [Étape 1 : extraction des PDF](#2-étape-1--extraction-des-pdf)
3. [Étape 2 : nettoyage et déduplication](#3-étape-2--nettoyage-et-déduplication)
4. [Étape 3 : chunking RAPTOR](#4-étape-3--chunking-raptor)
5. [Étape 4 : Contextual Retrieval](#5-étape-4--contextual-retrieval)
6. [Étape 5 : indexation](#6-étape-5--indexation)
7. [Étape 6 : cache sémantique](#7-étape-6--cache-sémantique)
8. [Étape 7 : répondre à une question](#8-étape-7--répondre-à-une-question)
9. [Étape 8 : évaluation](#9-étape-8--évaluation)

---

## 1. Principe d'un RAG

Un LLM seul peut inventer des chiffres. Un RAG l'oblige à s'appuyer sur des documents :

1. on **retrouve** les passages utiles dans un corpus ;
2. on les **injecte** dans le prompt du LLM ;
3. le LLM **rédige** une réponse à partir de ces passages uniquement, et cite ses sources.

Ici, le corpus est un ensemble de rapports statistiques officiels de Côte d'Ivoire, et le prompt système impose : répondre uniquement à partir du contexte, ne jamais inventer de chiffre, dire « Je n'ai pas cette information dans les documents disponibles » si le contexte est insuffisant, signaler les écarts entre documents, et terminer par les sources `[Source : Titre, Année, p.XX]`.

## 2. Étape 1 : extraction des PDF

_Script : `src/chunking.py`_

Les rapports contiennent du texte, mais surtout beaucoup de **tableaux** chiffrés. Le pipeline combine donc plusieurs moteurs :

| Besoin | Outil | Rôle |
|---|---|---|
| Texte | OpenDataLoader | Extraction principale, avec détection des blocs et des tableaux |
| Tableaux | pdfplumber + Camelot (modes *lattice* et *stream*) | Tableaux avec ou sans bordures, résultats fusionnés |
| Secours | PyMuPDF | Utilisé si l'extraction principale échoue ; détecte aussi les PDF « images » (scans) |

Les tableaux sont convertis en texte, et une **description générée par le LLM local** est associée à chacun pour qu'il soit retrouvable par une question en langage naturel.

## 3. Étape 2 : nettoyage et déduplication

Les PDF contiennent du bruit (en-têtes, pieds de page, tables des matières, questionnaires) qui pollue la recherche. Le pipeline applique des couches de filtrage successives :

- suppression des lignes d'en-tête et de pied de page répétées d'une page à l'autre ;
- détection des pages purement structurelles (couverture, sommaire…) ;
- détection des chunks de type questionnaire ou résumé exécutif ;
- détection des chunks « poubelle » ;
- **préservation des chiffres** : les règles de nettoyage ont été corrigées pour ne pas supprimer les valeurs numériques (important pour des statistiques) ;
- **déduplication** par similarité de Jaccard sur des « shingles » de 5 éléments, avec un seuil de 0,80.

## 4. Étape 3 : chunking RAPTOR

Un chunk trop petit perd le contexte, un chunk trop grand noie l'information. RAPTOR construit donc un **arbre à 4 niveaux** :

| Niveau | Contenu | Construction |
|---|---|---|
| Feuille (`leaf`) | Passage de ≈ 1 400 caractères (≈ 256 tokens), chevauchement de 200 | Découpage du texte |
| Section (`section`) | Résumé de 5 feuilles | Résumé LLM |
| Chapitre (`chapter`) | Résumé de 4 sections | Résumé LLM |
| Document (`document`) | Résumé global du rapport | Résumé LLM |

Bornes de taille : minimum 150 caractères, maximum 3 000 (pour ne pas couper les longs tableaux). L'intérêt : une question précise (« quel est le taux de pauvreté en 2018 ? ») trouve une feuille, tandis qu'une question de synthèse (« que contient ce rapport ? ») trouve un résumé de plus haut niveau.

Résultat sur le corpus : 15 164 chunks (12 608 feuilles, 2 035 sections, 490 chapitres, 31 documents), dont 4 519 issus de tableaux.

## 5. Étape 4 : Contextual Retrieval

Un passage isolé comme « Le taux est passé à 39,5 % » est ambigu. Avant l'indexation, le LLM génère pour chaque chunk une **courte description de contexte** (de quel document, de quelle enquête, de quelle section il provient). Le texte indexé est le contenu **enrichi** de cette description. Si le LLM est indisponible, un contexte minimal est construit à partir des métadonnées du document.

Chaque chunk reçoit aussi des métadonnées : année, localités, thèmes, série d'enquête (EHCVM, EDS, ENV, ENE, RGPH, MICS…), niveau RAPTOR, page, URL source, indicateur « tableau ».

Sortie : un fichier `*_chunks.json` par document dans `chunks_output_v2/`.

## 6. Étape 5 : indexation

> Le script qui réalise cette étape n'est pas inclus dans ce dépôt (voir README, section « À compléter »). Ce qui suit décrit les artefacts qu'il produit et que `rag_system.py` charge.

Trois index complémentaires sont construits dans `embeddings_v2/` :

| Index | Fichier | Rôle |
|---|---|---|
| Dense (sémantique) | `faiss_index.bin` + `embeddings.npz` | Embeddings **BGE-M3** (dim 1024), recherche par similarité vectorielle. Trouve le sens, même avec des mots différents. |
| Lexical | `bm25_index.pkl` | BM25. Trouve les termes exacts (sigles, noms de lieux, valeurs). |
| Graphe de métadonnées | `graph_index.json` | Associe à chaque chunk son année, ses localités, ses thèmes, sa série. Sert à favoriser les chunks qui correspondent au contexte de la question. |

`chunk_map.json` relie l'identifiant d'un chunk à son texte et à ses métadonnées.

**Pourquoi deux index ?** La recherche dense comprend la paraphrase mais rate parfois un sigle ou un nombre précis. BM25 fait l'inverse. Les combiner donne de meilleurs résultats que chacun seul.

## 7. Étape 6 : cache sémantique

_Script : `src/build_cache_semantique.py`_

Beaucoup de questions se ressemblent. Un index FAISS séparé est construit sur les **questions** d'un jeu de questions/réponses (`finetune_dataset.jsonl`, ≈ 81 000 paires annoncées). Les réponses du type « je n'ai pas cette information » sont exclues du cache : l'index final contient **69 198 entrées**.

À l'exécution, si une question a une similarité supérieure à **0,88** avec une question du cache, sa réponse est renvoyée immédiatement, sans appeler le LLM. Une vérification des mots-clés critiques (localité, thème, année, série) évite les faux positifs du type « rural » contre « rural et urbain ».

Un second cache, **conversationnel**, mémorise les 500 dernières questions de la session avec un seuil plus strict (0,97).

## 8. Étape 7 : répondre à une question

_Script : `src/rag_system.py`, méthode `ANSTATRagSystem.ask()`_

```
Question
  │
  ├─1─ Cache sémantique (sim > 0,88)        → réponse directe
  ├─2─ Cache conversationnel (sim > 0,97)   → réponse directe
  │
  ├─3─ Analyse de la question : années, localités, thèmes, séries, niveau RAPTOR souhaité
  ├─4─ Recherche dense FAISS (top 50)  +  recherche BM25 (top 50)
  ├─5─ Fusion RRF (Reciprocal Rank Fusion, k = 60, poids 0,6 dense / 0,4 BM25) → top 25
  ├─6─ Bonus du graphe (ajouté avec un coefficient 0,1)
  ├─7─ Reranker BGE (cross-encoder) sur le texte enrichi → top 5
  └─8─ Génération : prompt système + historique (6 tours) + 5 passages + question
        → Qwen2.5-7B-Instruct (vLLM), température 0,1, 800 tokens max
        → réponse + sources formatées
```

Détails utiles :

- **Étape 3, analyse de la question.** Des expressions régulières et des dictionnaires détectent les années citées (ou « récent » → 3 dernières années), les régions et zones (rural, urbain, nord, sud, Abidjan…), les thèmes (pauvreté, emploi, santé…) et les séries d'enquêtes. Les mots « résumé », « synthèse » orientent vers les niveaux supérieurs de l'arbre RAPTOR.
- **Étape 5, RRF.** Chaque passage reçoit un score `poids / (k + rang + 1)` dans chaque liste, puis les scores sont additionnés. Un passage bien classé dans les deux listes monte.
- **Étape 6, bonus du graphe.** Année correspondante : +3 (non correspondante : −0,3) ; chaque localité : +2 ; chaque thème : +1,5 ; chaque série : +2 ; niveau RAPTOR attendu : +0,5 ; tableau : +0,4. Le score combiné vaut `RRF + bonus × 0,1`.
- **Étape 7, reranking.** Un cross-encoder lit la question et chaque candidat ensemble, ce qui est plus précis (mais plus lent) que la comparaison de vecteurs. C'est pourquoi on ne l'applique qu'aux 25 candidats restants.
- **Mode dégradé.** Si le serveur vLLM est injoignable, le système renvoie une réponse extractive (les passages trouvés, sans rédaction).

## 9. Étape 8 : évaluation

_Scripts : `src/evaluation_ragas.py`, `src/evaluation_finetuned.py`_

1. `evaluation_ragas.py` tire un **échantillon aléatoire de 1 000 questions** (graine 42) dans le jeu de questions/réponses, appelle le vrai système RAG (cache désactivé) et enregistre la réponse et les passages utilisés, au format RAGAS. Un fichier de checkpoint permet de reprendre après une interruption.
2. `evaluation_finetuned.py` lance RAGAS sur ce jeu, par lots, avec reprise automatique si le serveur vLLM tombe.

Métriques RAGAS utilisées :

| Métrique | Question posée |
|---|---|
| Faithfulness | La réponse est-elle appuyée par les passages fournis (pas d'invention) ? |
| Context precision | Les passages récupérés sont-ils pertinents et bien classés ? |
| Context recall | Les passages récupérés contiennent-ils l'information nécessaire ? |
| Answer correctness | La réponse est-elle proche de la réponse de référence ? |
| Answer relevancy | La réponse répond-elle bien à la question ? (version de base uniquement) |

Le score global est une moyenne pondérée de ces métriques. Voir le [README](../README.md#3-résultats-dévaluation-ragas) pour les valeurs et les précautions de lecture.

> **Point méthodologique à vérifier avant de défendre le mémoire.** Le jeu de questions utilisé pour l'évaluation (`finetune_dataset.jsonl`) est le même fichier que celui annoncé pour le fine-tuning et pour le cache sémantique. Si les 1 000 questions évaluées ont aussi servi à entraîner le modèle fine-tuné, les scores de la version « RAG + fine-tuning » sont optimistes. Un jeu de test tenu à l'écart de l'entraînement est la façon propre de comparer les deux versions.
