# ============================================================================
# BUILD_CACHE_SEMANTIQUE.PY — Cache sémantique FAISS sur 81 000 Q/R
# ============================================================================
# Ce script est INDÉPENDANT du pipeline d'indexation existant.
# Il crée un index FAISS séparé dans embeddings_v2/cache_semantique/
#
# USAGE :
#   python build_cache_semantique.py
#
# SORTIES :
#   embeddings_v2/cache_semantique/
#     cache_faiss.bin    ← index FAISS des questions (dim=1024)
#     cache_map.json     ← {index → {question, answer, source}}
#     cache_meta.json    ← stats globales
#
# INTÉGRATION dans rag_system.py :
#   Avant le RAG → cherche dans ce cache
#   Si similarité > SEUIL → retourne la réponse directement
#   Sinon → RAG normal
# ============================================================================

import os
import json
import time
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import faiss

try:
    from FlagEmbedding import BGEM3FlagModel
    FLAG_AVAILABLE = True
    print("✅ FlagEmbedding actif")
except ImportError:
    FLAG_AVAILABLE = False
    from sentence_transformers import SentenceTransformer
    print("⚠️  FlagEmbedding non disponible — fallback SentenceTransformers")

# ============================================================================
# CONFIGURATION
# ============================================================================

JSONL_FILE   = "finetune_dataset.jsonl"       # ton fichier de 81 000 Q/R
OUTPUT_DIR   = Path("./embeddings_v2/cache_semantique")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FAISS_FILE = OUTPUT_DIR / "cache_faiss.bin"
CACHE_MAP_FILE   = OUTPUT_DIR / "cache_map.json"
CACHE_META_FILE  = OUTPUT_DIR / "cache_meta.json"

EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL_PATH", "BAAI/bge-m3")  # chemin local
BATCH_SIZE       = 32     # augmente si ta VRAM le permet (RTX 4060 = 32 ok)
SEUIL_SIMILARITE = 0.88   # seuil pour considérer une question comme connue

# Phrases à ignorer — on ne met pas les "je ne sais pas" dans le cache
PHRASES_INCONNUES = [
    "je n'ai pas cette information",
    "je ne dispose pas",
    "cette information n'est pas disponible",
    "je ne sais pas",
    "non disponible dans les documents",
]

# ============================================================================
# ÉTAPE 1 — LECTURE DU JSONL
# ============================================================================

def parse_line(line: str):
    """Extrait question + réponse depuis une ligne JSONL."""
    try:
        obj      = json.loads(line.strip())
        messages = obj.get("messages", [])

        user_msg      = next((m for m in messages if m["role"] == "user"),      None)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

        if not user_msg or not assistant_msg:
            return None

        user_content = user_msg["content"]
        answer       = assistant_msg["content"].strip()

        # Ignorer les réponses "je ne sais pas"
        answer_lower = answer.lower()
        if any(p in answer_lower for p in PHRASES_INCONNUES):
            return None

        # Extraire la question
        question = ""
        if "QUESTION :" in user_content:
            question = user_content.split("QUESTION :")[-1].strip()
        elif "QUESTION:" in user_content:
            question = user_content.split("QUESTION:")[-1].strip()
        else:
            lines    = [l.strip() for l in user_content.split("\n") if l.strip()]
            question = lines[-1] if lines else ""

        # Extraire la source depuis le contenu user
        source = ""
        for l in user_content.split("\n"):
            if l.strip().startswith("[Source"):
                source = l.strip()
                break

        if not question or not answer:
            return None

        return {
            "question": question,
            "answer":   answer,
            "source":   source,
        }

    except Exception:
        return None


def load_jsonl(path: str):
    """Charge et filtre toutes les Q/R du dataset."""
    print(f"\n📂 Lecture du dataset : {path}")

    if not Path(path).exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {path}")

    records = []
    errors  = 0
    ignored = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Lecture JSONL"):
            if not line.strip():
                continue
            parsed = parse_line(line)
            if parsed:
                records.append(parsed)
            else:
                # Distinguer erreurs vs ignorés (je ne sais pas)
                try:
                    obj = json.loads(line.strip())
                    msgs = obj.get("messages", [])
                    asst = next((m for m in msgs if m["role"] == "assistant"), None)
                    if asst:
                        answer_lower = asst["content"].lower()
                        if any(p in answer_lower for p in PHRASES_INCONNUES):
                            ignored += 1
                            continue
                except Exception:
                    pass
                errors += 1

    print(f"   ✅ {len(records):,} Q/R utiles chargées")
    print(f"   ⏭️  {ignored:,} Q/R 'je ne sais pas' ignorées")
    print(f"   ❌ {errors:,} erreurs de parsing")

    return records


# ============================================================================
# ÉTAPE 2 — CHARGEMENT DU MODÈLE D'EMBEDDING
# ============================================================================

def load_model():
    """Charge BGE-M3 depuis le chemin local."""
    print(f"\n🤖 Chargement BGE-M3 depuis : {EMBEDDING_MODEL}")

    if FLAG_AVAILABLE:
        model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
        print("   ✅ BGE-M3 via FlagEmbedding (GPU si disponible)")
    else:
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("   ✅ BGE-M3 via SentenceTransformers")

    return model


# ============================================================================
# ÉTAPE 3 — ENCODAGE DES QUESTIONS
# ============================================================================

def encode_questions(model, questions: list) -> np.ndarray:
    """Encode toutes les questions en vecteurs denses normalisés."""
    print(f"\n⚡ Encodage de {len(questions):,} questions...")
    t0 = time.time()

    all_embs = []

    if FLAG_AVAILABLE:
        # FlagEmbedding — batch manuel
        for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="Encodage BGE-M3"):
            batch = questions[i:i + BATCH_SIZE]
            out   = model.encode(
                batch,
                batch_size       = BATCH_SIZE,
                max_length       = 256,        # questions courtes → 256 suffit
                return_dense     = True,
                return_sparse    = False,
                return_colbert_vecs = False,
            )
            embs  = out["dense_vecs"].astype(np.float32)
            # Normalisation L2 → cosine similarity via Inner Product
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs  = embs / np.maximum(norms, 1e-9)
            all_embs.append(embs)
    else:
        # SentenceTransformers
        embs = model.encode(
            questions,
            batch_size          = BATCH_SIZE,
            show_progress_bar   = True,
            normalize_embeddings= True,
            convert_to_numpy    = True,
        ).astype(np.float32)
        all_embs.append(embs)

    embeddings = np.vstack(all_embs)
    elapsed    = time.time() - t0

    print(f"   ✅ {embeddings.shape[0]:,} vecteurs dim={embeddings.shape[1]}")
    print(f"   ⏱️  Durée : {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return embeddings


# ============================================================================
# ÉTAPE 4 — CONSTRUCTION DE L'INDEX FAISS
# ============================================================================

def build_faiss_cache(embeddings: np.ndarray) -> faiss.Index:
    """Construit l'index FAISS pour le cache sémantique."""
    n, dim = embeddings.shape
    print(f"\n🗂️  Construction index FAISS — {n:,} vecteurs dim={dim}")

    if n < 10_000:
        # Petit corpus → FlatIP exact
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"   📦 FAISS FlatIP (exact)")
    else:
        # Grand corpus → IVFFlat approximé
        nlist     = min(512, n // 100)
        quantizer = faiss.IndexFlatIP(dim)
        index     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add_with_ids(embeddings, np.arange(n, dtype=np.int64))
        index.nprobe = 32
        index.make_direct_map()
        print(f"   📦 FAISS IVFFlat — nlist={nlist}, nprobe=32")

    return index


# ============================================================================
# ÉTAPE 5 — SAUVEGARDE
# ============================================================================

def save_cache(index, records, embeddings):
    """Sauvegarde l'index FAISS et la map question→réponse."""
    print(f"\n💾 Sauvegarde dans {OUTPUT_DIR}...")

    # Index FAISS
    faiss.write_index(index, str(CACHE_FAISS_FILE))
    print(f"   ✅ {CACHE_FAISS_FILE}")

    # Map index → {question, answer, source}
    cache_map = {}
    for i, rec in enumerate(records):
        cache_map[str(i)] = {
            "question": rec["question"],
            "answer":   rec["answer"],
            "source":   rec["source"],
        }

    with open(CACHE_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_map, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {CACHE_MAP_FILE}")

    # Métadonnées
    meta = {
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_entries":  len(records),
        "embedding_dim":  int(embeddings.shape[1]),
        "embedding_model": EMBEDDING_MODEL,
        "seuil_similarite": SEUIL_SIMILARITE,
        "batch_size":     BATCH_SIZE,
    }
    with open(CACHE_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {CACHE_META_FILE}")

    print(f"\n   📊 Résumé cache :")
    print(f"      Q/R indexées  : {len(records):,}")
    print(f"      Dim vecteur   : {embeddings.shape[1]}")
    print(f"      Seuil         : {SEUIL_SIMILARITE}")


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    print("\n" + "=" * 65)
    print("  BUILD CACHE SÉMANTIQUE — ANSTAT-IA")
    print("  BGE-M3 + FAISS sur 81 000 Q/R")
    print("=" * 65)

    t_total = time.time()

    # Étape 1 — Charger les Q/R
    records = load_jsonl(JSONL_FILE)
    if not records:
        print("❌ Aucune Q/R chargée !")
        return

    # Étape 2 — Charger le modèle
    model = load_model()

    # Étape 3 — Encoder les questions
    questions  = [r["question"] for r in records]
    embeddings = encode_questions(model, questions)

    # Étape 4 — Construire l'index FAISS
    index = build_faiss_cache(embeddings)

    # Étape 5 — Sauvegarder
    save_cache(index, records, embeddings)

    elapsed = time.time() - t_total
    print(f"\n🎉 Cache sémantique construit en {elapsed/60:.1f} minutes !")
    print(f"📁 Fichiers dans : {OUTPUT_DIR}")
    print(f"\n➡️  Prochaine étape : intégrer dans rag_system.py")
    print(f"   Chercher dans le cache AVANT le RAG")
    print(f"   Seuil de similarité : {SEUIL_SIMILARITE}")


if __name__ == "__main__":
    main()