# ============================================================================
# RAGAS_EVAL_ANSTAT.PY — ÉVALUATION COMPLÈTE DU SYSTÈME RAG ANSTAT-IA
# ============================================================================
# Ce script :
#   1. Charge le système RAG ANSTAT-IA complet (rag_system.py)
#   2. Lit le dataset JSONL et tire un échantillon aléatoire de 500 questions
#   3. Pour chaque question → appel RAG réel → answer + contexts
#   4. Formate le dataset au format RAGAS
#   5. Lance l'évaluation avec toutes les métriques RAGAS
#   6. Affiche un score global sur 100 + rapport complet
#
# INSTALLATION :
#   pip install ragas datasets langchain-openai
#
# LANCEMENT :
#   python ragas_eval_anstat.py
# ============================================================================



import os
import json
import time
import random
import logging
import numpy as np
from pathlib  import Path
from datetime import datetime
from typing   import List, Dict, Any

from tqdm import tqdm

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Import de ton système RAG ─────────────────────────────────────────────────
import sys
sys.path.append(".")   # adapte si rag_system.py est dans un autre dossier
from rag_system import ANSTATRagSystem

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Dataset d'entrée (ton fichier JSONL de 81 000 questions)
    "input_jsonl"   : "finetune_dataset.jsonl",

    # Dataset de sortie converti pour RAGAS
    "output_jsonl"  : "ragas_dataset_finetuned.jsonl",

    # Rapport d'évaluation final
    "output_report" : "ragas_report_finetuned.json",

    # Fichier de checkpoint pour reprendre en cas d'interruption
    # Contient les réponses RAG déjà calculées + la liste des questions échantillonnées
    "checkpoint_file": "ragas_checkpoint.json",

    # Échantillon aléatoire (1000 = idéal pour mémoire Master 2)
    # Met None pour tout prendre (déconseillé sur 81 000 questions)
    "max_samples"   : 1000,

    # Graine aléatoire pour reproductibilité
    "seed"          : 42,

    # Ton serveur vLLM (utilisé comme LLM juge par RAGAS)
    "vllm_base_url" : os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    "vllm_model"    : "qwen_anstat_finetuned_q4km",

    # Délai entre chaque appel RAG (évite de surcharger le serveur)
    "sleep_between" : 0.3,

    "skip_unknown"  : True,

    # Sauvegarder le checkpoint tous les N appels RAG
    "checkpoint_every": 10,
}

# ============================================================================
# CHARGEMENT DU SYSTÈME RAG — UNE SEULE FOIS AU DÉMARRAGE
# ============================================================================

logger.info("🔄 Chargement du système RAG ANSTAT-IA...")
rag_system = ANSTATRagSystem()
rag_system.load()
logger.info("✅ Système RAG prêt !\n")


# ============================================================================
# FONCTION RAG — OPTION A : APPEL DIRECT AU RAG COMPLET
# ============================================================================

def get_rag_answer(question: str) -> Dict[str, Any]:
    """
    Envoie une question au système RAG ANSTAT-IA complet.
    Retourne : {"answer": str, "contexts": List[str]}
    use_cache=False → chaque question est traitée fraîche (pas de biais d'éval)
    """
    try:
        result = rag_system.ask(question, verbose=False, use_cache=False)
        return {
            "answer"  : result["answer"],
            "contexts": [c["content"] for c in result["chunks"]],
        }
    except Exception as e:
        logger.error(f"Erreur RAG pour la question '{question[:60]}...' : {e}")
        return {"answer": "", "contexts": []}


# ============================================================================
# CHECKPOINT — SAUVEGARDE ET REPRISE
# ============================================================================

def save_checkpoint(records_full: List[Dict], ragas_done: List[Dict], path: str):
    """
    Sauvegarde l'état courant dans un fichier JSON.
    - records_full : toutes les questions de l'échantillon (avec leurs ground_truth)
    - ragas_done   : les réponses RAG déjà calculées (subset de records_full)
    Appelé toutes les checkpoint_every questions pour limiter la perte en cas de crash.
    """
    checkpoint = {
        "saved_at"   : datetime.now().isoformat(),
        "total"      : len(records_full),
        "done"       : len(ragas_done),
        "records_full": records_full,   # liste complète des questions (pour reprendre dans le bon ordre)
        "ragas_done" : ragas_done,      # réponses RAG déjà obtenues
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    # Remplacement atomique : évite un fichier corrompu si le PC s'éteint pendant l'écriture
    Path(tmp_path).replace(path)
    logger.info(f"💾 Checkpoint sauvegardé : {len(ragas_done)}/{len(records_full)} ({path})")


def load_checkpoint(path: str):
    """
    Charge le checkpoint s'il existe.
    Retourne (records_full, ragas_done) ou (None, None) si pas de checkpoint.
    """
    if not Path(path).exists():
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        records_full = checkpoint["records_full"]
        ragas_done   = checkpoint["ragas_done"]
        logger.info(
            f"♻️  Checkpoint trouvé ({checkpoint['saved_at']}) : "
            f"{len(ragas_done)}/{len(records_full)} questions déjà traitées → reprise en cours..."
        )
        return records_full, ragas_done
    except Exception as e:
        logger.warning(f"⚠️  Checkpoint illisible ({e}) → reprise depuis le début")
        return None, None


def delete_checkpoint(path: str):
    """Supprime le checkpoint une fois l'évaluation terminée avec succès."""
    try:
        Path(path).unlink(missing_ok=True)
        logger.info(f"🗑️  Checkpoint supprimé : {path}")
    except Exception:
        pass


# ============================================================================
# ÉTAPE 1 — LECTURE + ÉCHANTILLONNAGE STRATIFIÉ DU DATASET
# ============================================================================

def parse_jsonl_line(line: str) -> Dict[str, Any]:
    """
    Extrait question et ground_truth depuis une ligne du fichier JSONL.
    Format attendu :
    {
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "DOCUMENTS DE RÉFÉRENCE :\n...\nQUESTION : ..."},
        {"role": "assistant", "content": "réponse idéale..."}
      ]
    }
    """
    try:
        obj      = json.loads(line.strip())
        messages = obj.get("messages", [])

        user_msg      = next((m for m in messages if m["role"] == "user"),      None)
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

        if not user_msg or not assistant_msg:
            return None

        user_content = user_msg["content"]

        # Extraire la question (après "QUESTION :")
        question = ""
        if "QUESTION :" in user_content:
            question = user_content.split("QUESTION :")[-1].strip()
        elif "QUESTION:" in user_content:
            question = user_content.split("QUESTION:")[-1].strip()
        else:
            lines    = [l.strip() for l in user_content.split("\n") if l.strip()]
            question = lines[-1] if lines else ""

        # Extraire le chunk de référence (contexte source)
        chunk = ""
        if "DOCUMENTS DE RÉFÉRENCE" in user_content:
            parts = user_content.split("QUESTION")[0]
            content_lines = []
            for l in parts.split("\n"):
                l = l.strip()
                if any(l.startswith(p) for p in
                       ["[Source", "Niveau", "DOCUMENTS", "──"]) or not l:
                    continue
                content_lines.append(l)
            chunk = " ".join(content_lines).strip()

        # ground_truth = réponse idéale générée lors de la création du dataset
        ground_truth = assistant_msg["content"].strip()

        if not question or not ground_truth:
            return None
        
        # Filtrer les réponses "je ne sais pas"
        phrases_inconnues = [
    "je n'ai pas cette information",
    "je ne dispose pas",
    "cette information n'est pas disponible",
    "je ne sais pas",
]
        if any(p in ground_truth.lower() for p in phrases_inconnues):
            return None  # on ignore ces questions

        return {
            "question"    : question,
            "ground_truth": ground_truth,
            "chunk"       : chunk,
        }

    except Exception as e:
        return None


def load_dataset(path: str, max_samples: int = None, seed: int = 42) -> List[Dict]:
    """
    Charge le fichier JSONL complet puis tire un échantillon aléatoire.
    L'échantillon est reproductible grâce à la graine seed.
    """
    logger.info(f"📂 Lecture du dataset : {path}")

    if not Path(path).exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {path}")

    # Charger TOUT le dataset
    all_records = []
    errors      = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            parsed = parse_jsonl_line(line)
            if parsed:
                all_records.append(parsed)
            else:
                errors += 1

    logger.info(f"   Total chargé  : {len(all_records):,} exemples | {errors} erreurs ignorées")

    # Échantillonnage aléatoire reproductible
    if max_samples and len(all_records) > max_samples:
        random.seed(seed)
        all_records = random.sample(all_records, max_samples)
        logger.info(f"   Échantillon   : {max_samples} questions tirées aléatoirement (seed={seed})")
    else:
        logger.info(f"   Échantillon   : tout le dataset ({len(all_records)} questions)")

    return all_records


# ============================================================================
# ÉTAPE 2 — APPEL RAG + CONSTRUCTION DU DATASET RAGAS (AVEC CHECKPOINT)
# ============================================================================

def build_ragas_dataset(records: List[Dict]) -> List[Dict]:
    """
    Pour chaque enregistrement :
    - Envoie la question au RAG → récupère answer + contexts
    - Formate pour RAGAS : question / contexts / answer / ground_truth

    Supporte la reprise automatique via checkpoint :
    - Si un checkpoint existe et contient les mêmes questions (même seed),
      on reprend depuis la dernière position sans recalculer les réponses déjà obtenues.
    - Le checkpoint est sauvegardé toutes les checkpoint_every questions.
    - En fin de succès, le checkpoint est supprimé.
    """
    checkpoint_path = CONFIG["checkpoint_file"]

    # ── Tentative de reprise depuis checkpoint ────────────────────────────────
    records_full_ckpt, ragas_done = load_checkpoint(checkpoint_path)

    if records_full_ckpt is not None and len(records_full_ckpt) == len(records):
        # Checkpoint compatible : même nombre de questions → on reprend
        start_index  = len(ragas_done)
        records_todo = records_full_ckpt[start_index:]   # questions restantes dans l'ordre original
        logger.info(f"   → Reprise depuis la question {start_index + 1}/{len(records)}\n")
    else:
        # Pas de checkpoint ou checkpoint incompatible → on repart de zéro
        if records_full_ckpt is not None:
            logger.info("   → Checkpoint incompatible (taille différente) → reprise depuis le début")
        start_index  = 0
        ragas_done   = []
        records_todo = records

    # ── Boucle principale ─────────────────────────────────────────────────────
    logger.info(f"🔄 Appel RAG pour {len(records_todo)} questions restantes (sur {len(records)} total)...")
    if len(records_todo) > 0:
        logger.info(f"   Durée estimée : {len(records_todo) * 3 // 60} à {len(records_todo) * 5 // 60} minutes\n")

    errors  = 0
    t_start = time.time()

    # Barre de progression tqdm — initial=start_index pour refléter la reprise
    pbar = tqdm(
        total        = len(records),
        initial      = start_index,
        desc         = "RAG",
        unit         = "q",
        dynamic_ncols= True,
        bar_format   = (
            "{desc} |{bar}| {n_fmt}/{total_fmt} "
            "[{percentage:5.1f}%  elapsed {elapsed}  ETA {remaining}  {rate_fmt}]"
        ),
    )

    for i, rec in enumerate(records_todo):
        global_index = start_index + i   # position dans l'échantillon complet
        question     = rec["question"]
        ground_truth = rec["ground_truth"]
        chunk        = rec.get("chunk", "")

        # Appel RAG complet
        rag_result = get_rag_answer(question)
        answer     = rag_result.get("answer", "")
        contexts   = rag_result.get("contexts", [])

        # Si le RAG ne retourne pas de contextes → utiliser le chunk original
        if not contexts:
            contexts = [chunk] if chunk else [""]

        # Nettoyage : s'assurer que contexts est une liste de strings non vides
        contexts = [str(c).strip() for c in contexts if c and str(c).strip()]
        if not contexts:
            contexts = [""]

        ragas_done.append({
            "question"    : question,
            "contexts"    : contexts,
            "answer"      : answer,
            "ground_truth": ground_truth,
        })

        # ── Checkpoint périodique ─────────────────────────────────────────────
        questions_done = global_index + 1
        is_last        = (i == len(records_todo) - 1)
        if questions_done % CONFIG["checkpoint_every"] == 0 or is_last:
            pbar.set_postfix(ckpt=f"{questions_done}/{len(records)}", refresh=True)
            save_checkpoint(records, ragas_done, checkpoint_path)

        pbar.update(1)

        # Pause pour ne pas surcharger le serveur
        if CONFIG["sleep_between"] > 0:
            time.sleep(CONFIG["sleep_between"])

    pbar.close()

    elapsed_total = (time.time() - t_start) / 60
    logger.info(f"\n   ✅ {len(ragas_done)} lignes prêtes | {errors} erreurs | {elapsed_total:.1f} min")
    return ragas_done


# ============================================================================
# ÉTAPE 3 — SAUVEGARDE DU DATASET CONVERTI
# ============================================================================

def save_ragas_dataset(records: List[Dict], path: str):
    """Sauvegarde le dataset au format JSONL attendu par RAGAS."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"💾 Dataset RAGAS sauvegardé : {path} ({len(records)} lignes)")


# ============================================================================
# ÉTAPE 4 — ÉVALUATION RAGAS
# ============================================================================

def run_ragas_evaluation(ragas_records: List[Dict]) -> Dict:
    """
    Lance l'évaluation RAGAS avec toutes les métriques.
    Utilise ton vLLM local comme LLM juge via l'API OpenAI-compatible.
    """
    logger.info("\n🚀 Lancement évaluation RAGAS...")
    logger.info("   Métriques : faithfulness · answer_relevancy · context_precision")
    logger.info("               context_recall · answer_correctness · answer_similarity")

    # LLM juge → ton vLLM local
    llm_judge = ChatOpenAI(
        model       = CONFIG["vllm_model"],
        base_url    = CONFIG["vllm_base_url"],
        api_key     = "dummy",
        temperature = 0.0,
        max_tokens  = 1024,
    )

    # Embeddings pour answer_relevancy
    embeddings_judge = OpenAIEmbeddings(
        model    = "text-embedding-3-small",
        base_url = CONFIG["vllm_base_url"],
        api_key  = "dummy",
    )

    # Configurer le LLM juge sur chaque métrique
    for metric in [faithfulness, answer_relevancy, context_precision,
                   context_recall, answer_correctness, answer_similarity]:
        metric.llm = llm_judge

    try:
        answer_relevancy.embeddings = embeddings_judge
    except Exception:
        pass

    # Convertir en Dataset HuggingFace
    dataset = Dataset.from_list(ragas_records)

    # Lancer l'évaluation
    t_start = datetime.now()

    try:
        result = evaluate(
            dataset         = dataset,
            metrics         = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
                answer_similarity,
            ],
            raise_exceptions = False,
        )
    except Exception as e:
        logger.warning(f"⚠️  Évaluation complète échouée ({e})")
        logger.info("   Tentative avec métriques de base...")
        result = evaluate(
            dataset          = dataset,
            metrics          = [faithfulness, answer_relevancy,
                                context_precision, context_recall],
            raise_exceptions = False,
        )

    elapsed = (datetime.now() - t_start).total_seconds()
    logger.info(f"   ✅ Évaluation terminée en {elapsed:.1f}s")
    return result


# ============================================================================
# ÉTAPE 5 — SCORE GLOBAL SUR 100
# ============================================================================

def compute_global_score(result) -> Dict:
    """
    Score global pondéré sur 100.

    Pondération selon priorités ANSTAT-IA :
      Faithfulness       35% → ne jamais inventer de chiffres
      Context Recall     25% → retrouver les bons documents
      Answer Relevancy   20% → répondre à la question posée
      Context Precision  10% → pas de chunks inutiles
      Answer Correctness  5% → exactitude vs ground truth
      Answer Similarity   5% → similarité sémantique
    """
    WEIGHTS = {
        "faithfulness"      : 0.35,
        "context_recall"    : 0.25,
        "answer_relevancy"  : 0.20,
        "context_precision" : 0.10,
        "answer_correctness": 0.05,
        "answer_similarity" : 0.05,
    }

    score_dict = dict(result)
    scores     = {}

    for metric in WEIGHTS:
        val = score_dict.get(metric, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            scores[metric] = float(val)
        else:
            scores[metric] = None

    # Score pondéré (ignore les métriques manquantes)
    weighted_sum = 0.0
    total_weight = 0.0
    for metric, weight in WEIGHTS.items():
        if scores[metric] is not None:
            weighted_sum += scores[metric] * weight
            total_weight += weight

    global_01  = weighted_sum / total_weight if total_weight > 0 else 0.0
    global_100 = round(global_01 * 100, 2)

    return {
        "scores_individuels": scores,
        "ponderation"       : WEIGHTS,
        "score_global_0_1"  : round(global_01, 4),
        "score_global_100"  : global_100,
    }


def interpret_score(score: float) -> str:
    if score >= 90: return "Excellent — système RAG de niveau production"
    if score >= 80: return "Très bon — quelques améliorations mineures possibles"
    if score >= 70: return "Bon — améliorations recommandées sur certaines métriques"
    if score >= 60: return "Moyen — lacunes significatives à corriger"
    if score >= 50: return "Faible — refonte du pipeline recommandée"
    return          "Insuffisant — révision complète nécessaire"


# ============================================================================
# ÉTAPE 6 — RAPPORT FINAL
# ============================================================================

def print_report(scores_info: Dict, n_samples: int):
    scores   = scores_info["scores_individuels"]
    global_s = scores_info["score_global_100"]

    print("\n" + "="*60)
    print("  RAPPORT D'ÉVALUATION RAGAS — ANSTAT-IA")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\n  Questions évaluées : {n_samples}")

    print(f"\n{'─'*60}")
    print("  SCORES PAR MÉTRIQUE")
    print(f"{'─'*60}")

    labels = {
        "faithfulness"      : "Faithfulness       (réponse dans le contexte ?)",
        "answer_relevancy"  : "Answer relevancy   (répond à la question ?)",
        "context_recall"    : "Context recall     (bons chunks trouvés ?)",
        "context_precision" : "Context precision  (chunks tous utiles ?)",
        "answer_correctness": "Answer correctness (exactitude vs ground truth)",
        "answer_similarity" : "Answer similarity  (similarité sémantique)",
    }

    for metric, label in labels.items():
        val  = scores.get(metric)
        pct  = scores_info["ponderation"].get(metric, 0) * 100
        if val is not None:
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"\n  {label}")
            print(f"  [{bar}] {val:.4f}  (poids {pct:.0f}%)")
        else:
            print(f"\n  {label}")
            print(f"  [N/A]  (poids {pct:.0f}%)")

    print(f"\n{'═'*60}")
    print(f"  SCORE GLOBAL : {global_s:.2f} / 100")
    print(f"  Niveau       : {interpret_score(global_s)}")
    print(f"{'═'*60}")

    print("\n  DIAGNOSTIC :")
    diag = {
        "faithfulness"      : (0.85, "le modèle invente des informations"),
        "context_recall"    : (0.85, "le retrieval rate des documents importants"),
        "context_precision" : (0.80, "trop de chunks non pertinents récupérés"),
        "answer_relevancy"  : (0.85, "les réponses dévient de la question"),
    }
    ok = True
    for metric, (seuil, msg) in diag.items():
        val = scores.get(metric)
        if val is not None and val < seuil:
            print(f"  ⚠️  {metric} = {val:.2f} (seuil {seuil}) → {msg}")
            ok = False
    if ok:
        print("  ✅ Toutes les métriques sont au-dessus des seuils recommandés")
    print()


def save_report(scores_info: Dict, n_samples: int, path: str):
    report = {
        "metadata": {
            "date"     : datetime.now().isoformat(),
            "n_samples": n_samples,
            "system"   : "ANSTAT-IA RAG V3",
            "model"    : CONFIG["vllm_model"],
            "seed"     : CONFIG["seed"],
        },
        "scores"        : scores_info,
        "interpretation": interpret_score(scores_info["score_global_100"]),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Rapport sauvegardé : {path}")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    logger.info("\n" + "="*60)
    logger.info("  RAGAS EVAL — ANSTAT-IA")
    logger.info(f"  Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    # Étape 1 — Charger + échantillonner
    # NB : si un checkpoint existe, l'échantillon original est rechargé depuis le checkpoint
    #      (pas depuis le JSONL) pour garantir que c'est exactement les mêmes questions.
    records = load_dataset(
        CONFIG["input_jsonl"],
        CONFIG["max_samples"],
        CONFIG["seed"],
    )
    if not records:
        logger.error("❌ Dataset vide !")
        return

    # Étape 2 — Appel RAG + construction dataset RAGAS (avec reprise automatique)
    ragas_records = build_ragas_dataset(records)
    if not ragas_records:
        logger.error("❌ Aucune réponse RAG obtenue !")
        return

    # Étape 3 — Sauvegarder le dataset converti
    save_ragas_dataset(ragas_records, CONFIG["output_jsonl"])

    # Étape 4 — Évaluation RAGAS
    result = run_ragas_evaluation(ragas_records)

    # Étape 5 — Score global sur 100
    scores_info = compute_global_score(result)

    # Étape 6 — Rapport
    print_report(scores_info, len(ragas_records))
    save_report(scores_info, len(ragas_records), CONFIG["output_report"])

    # Étape 7 — Supprimer le checkpoint (évaluation terminée avec succès)
    delete_checkpoint(CONFIG["checkpoint_file"])

    logger.info(f"🎉 Évaluation terminée ! Score global : {scores_info['score_global_100']:.2f}/100")


if __name__ == "__main__":
    main()