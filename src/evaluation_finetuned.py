# ============================================================================
# RAGAS_EVAL_ROBUST.PY — ÉVALUATION RAGAS ANTI-CRASH TOTALE — VERSION CORRIGÉE
# ============================================================================

import os, json, time, logging, traceback, math
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
from pathlib  import Path
from datetime import datetime
from typing   import List, Dict, Any, Optional

os.environ.setdefault("OPENAI_API_KEY", "dummy-key-vllm-local")

import requests
from datasets        import Dataset
from ragas           import evaluate
from ragas.metrics   import (
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ragas_eval_robust.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "input_jsonl"     : "ragas_dataset_finetuned.jsonl",
    "output_report"   : "ragas_report_finetuned.json",
    "checkpoint_file" : "ragas_eval_checkpoint.json",
    "vllm_base_url"   : os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    "vllm_model"      : "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "batch_size"      : 20,
    "ragas_workers"   : 8,
    "request_timeout" : 300,
    "max_retries_llm" : 2,
    "batch_retry_count"   : 5,
    "batch_retry_delay_s" : 30,
    "connectivity_check_interval": 10,
    "connectivity_timeout"        : 10,
}

METRICS_TO_USE = [
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness,
]

METRIC_NAMES = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

WEIGHTS = {
    "faithfulness"      : 0.40,
    "context_recall"    : 0.30,
    "context_precision" : 0.20,
    "answer_correctness": 0.10,
}

# ============================================================================
# UTILITAIRES
# ============================================================================

def wait_for_server(base_url: str, max_wait_s: int = 300) -> bool:
    health_url  = base_url.rstrip("/").replace("/v1", "") + "/health"
    models_url  = base_url.rstrip("/") + "/models"
    deadline    = time.time() + max_wait_s
    attempt     = 0
    logger.info(f"⏳ Attente du serveur vLLM : {base_url}")
    while time.time() < deadline:
        attempt += 1
        for url in [health_url, models_url]:
            try:
                r = requests.get(url, timeout=CONFIG["connectivity_timeout"])
                if r.status_code < 500:
                    logger.info(f"   ✅ Serveur disponible après {attempt} tentative(s)")
                    return True
            except Exception:
                pass
        wait = min(10 * attempt, 60)
        logger.warning(f"   🔴 Serveur non disponible, nouvel essai dans {wait}s")
        time.sleep(wait)
    logger.error(f"❌ Serveur toujours indisponible après {max_wait_s}s")
    return False

def is_server_up(base_url: str) -> bool:
    health_url = base_url.rstrip("/").replace("/v1", "") + "/health"
    models_url = base_url.rstrip("/") + "/models"
    for url in [health_url, models_url]:
        try:
            r = requests.get(url, timeout=CONFIG["connectivity_timeout"])
            if r.status_code < 500:
                return True
        except Exception:
            pass
    return False

def make_llm():
    return ChatOpenAI(
        model        = CONFIG["vllm_model"],
        base_url     = CONFIG["vllm_base_url"],
        api_key      = "dummy-key-vllm-local",
        temperature  = 0.0,
        max_tokens   = 512,
        timeout      = CONFIG["request_timeout"],
        max_retries  = CONFIG["max_retries_llm"],
    )

def load_ragas_jsonl(path: str) -> List[Dict]:
    logger.info(f"📂 Chargement : {path}")
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {path}")
    records, errors = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj          = json.loads(line)
                question     = str(obj.get("question", "")).strip()
                answer       = str(obj.get("answer", "")).strip()
                ground_truth = str(obj.get("ground_truth", "")).strip()
                contexts     = obj.get("contexts", [])
                if not question or not answer or not ground_truth:
                    errors += 1
                    continue
                if isinstance(contexts, str):
                    contexts = [contexts]
                contexts = [str(c).strip() for c in contexts if str(c).strip()]
                if not contexts:
                    contexts = ["contexte non disponible"]
                records.append({
                    "question": question,
                    "contexts": contexts,
                    "answer": answer,
                    "ground_truth": ground_truth,
                })
            except Exception:
                errors += 1
    logger.info(f"   ✅ {len(records):,} exemples valides | {errors} erreurs ignorées")
    return records

def save_checkpoint(batch_index: int, total_batches: int, accumulated: Dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "saved_at": datetime.now().isoformat(),
            "batch_index": batch_index,
            "total_batches": total_batches,
            "accumulated": accumulated,
        }, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(path)
    logger.info(f"💾 Checkpoint : batch {batch_index}/{total_batches}")

def load_checkpoint(path: str):
    if not Path(path).exists():
        return 0, None, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            ck = json.load(f)
        logger.info(f"♻️  Checkpoint trouvé : batch {ck['batch_index']}/{ck['total_batches']} → reprise...")
        return ck["batch_index"], ck["total_batches"], ck["accumulated"]
    except Exception:
        return 0, None, {}

def delete_checkpoint(path: str):
    Path(path).unlink(missing_ok=True)

# ============================================================================
# ÉVALUATION D'UN BATCH — VERSION CORRIGÉE
# ============================================================================

def evaluate_one_batch(batch: List[Dict], llm_judge) -> Optional[Dict]:
    dataset = Dataset.from_list(batch)

    for attempt in range(1, CONFIG["batch_retry_count"] + 1):

        if not is_server_up(CONFIG["vllm_base_url"]):
            logger.warning(f"   🔴 Serveur down — tentative {attempt} — attente reconnexion...")
            if not wait_for_server(CONFIG["vllm_base_url"], max_wait_s=600):
                return None
            llm_judge = make_llm()
            for m in METRICS_TO_USE:
                m.llm = llm_judge

        try:
            logger.info(f"   ▶  Tentative {attempt}/{CONFIG['batch_retry_count']}...")
            t0 = time.time()

            result = evaluate(
                dataset          = dataset,
                metrics          = METRICS_TO_USE,
                llm              = llm_judge,
                raise_exceptions = False,
                run_config       = RunConfig(max_workers=CONFIG["ragas_workers"], timeout=CONFIG["request_timeout"]),
            )

            # ✅ Convertir en DataFrame et extraire les moyennes
            df = result.to_pandas()
            
            result_dict = {}
            for metric_name in METRIC_NAMES:
                if metric_name in df.columns:
                    vals = df[metric_name].dropna()
                    if len(vals) > 0:
                        result_dict[metric_name] = float(vals.mean())
                    else:
                        result_dict[metric_name] = None
                else:
                    result_dict[metric_name] = None

            has_valid = any(v is not None for v in result_dict.values())
            if not has_valid:
                logger.warning("   ⚠️  Aucune métrique valide dans ce batch")
                return None

            elapsed = time.time() - t0
            logger.info(f"   ✅ Batch OK en {elapsed:.0f}s")
            logger.info(f"   📊 Scores: {result_dict}")
            return result_dict

        except Exception as e:
            logger.warning(f"   ⚠️  Tentative {attempt} échouée : {type(e).__name__}: {e}")
            if attempt < CONFIG["batch_retry_count"]:
                delay = CONFIG["batch_retry_delay_s"] * attempt
                time.sleep(delay)

    logger.error(f"   ❌ Batch échoué après {CONFIG['batch_retry_count']} tentatives")
    return None

# ============================================================================
# ÉVALUATION PAR BATCHS
# ============================================================================

def evaluate_in_batches(records: List[Dict]) -> Dict[str, List]:
    ckpt_path  = CONFIG["checkpoint_file"]
    batch_size = CONFIG["batch_size"]
    batches    = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    total      = len(batches)

    start_batch, _, accumulated = load_checkpoint(ckpt_path)
    if not accumulated:
        accumulated = {name: [] for name in METRIC_NAMES}

    if not wait_for_server(CONFIG["vllm_base_url"], max_wait_s=300):
        raise ConnectionError("Serveur vLLM inaccessible")

    logger.info(f"\n🔌 Connexion vLLM : {CONFIG['vllm_base_url']}")
    llm_judge = make_llm()
    for m in METRICS_TO_USE:
        m.llm = llm_judge

    logger.info(f"\n🚀 Évaluation : {len(records):,} questions | {total} batchs")
    logger.info(f"   Métriques : {METRIC_NAMES}")

    t_global = time.time()
    skipped = 0

    for b_idx in range(start_batch, total):
        batch = batches[b_idx]
        logger.info(f"\n{'─'*50}\n  Batch {b_idx+1}/{total} | {len(batch)} questions\n{'─'*50}")

        t_batch = time.time()
        result_dict = evaluate_one_batch(batch, llm_judge)

        if result_dict:
            for name in METRIC_NAMES:
                accumulated[name].append(result_dict.get(name))
        else:
            skipped += 1
            for name in METRIC_NAMES:
                accumulated[name].append(None)

        # Affichage des scores cumulés
        print(f"\n  SCORES CUMULÉS (batchs 1→{b_idx+1}/{total}):")
        for name in METRIC_NAMES:
            vals = [v for v in accumulated[name] if v is not None]
            if vals:
                avg = np.mean(vals)
                bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
                print(f"    {name:22s} [{bar}] {avg*100:5.1f}% (n={len(vals)})")
            else:
                print(f"    {name:22s} [N/A]")

        elapsed_b = time.time() - t_batch
        remaining = total - b_idx - 1
        eta = (elapsed_b * remaining) / 60 if remaining > 0 else 0
        logger.info(f"  ⏱ Batch: {elapsed_b:.0f}s | ETA: {eta:.1f} min")

        save_checkpoint(b_idx + 1, total, accumulated, ckpt_path)

    logger.info(f"\n✅ Évaluation terminée | {skipped} batchs skippés")
    return accumulated

# ============================================================================
# SCORE GLOBAL ET RAPPORT
# ============================================================================

def compute_global_score(accumulated: Dict) -> Dict:
    scores = {}
    for name in METRIC_NAMES:
        vals = [v for v in accumulated.get(name, []) if v is not None]
        scores[name] = float(np.mean(vals)) if vals else None

    weighted_sum = 0.0
    total_weight = 0.0
    for metric, weight in WEIGHTS.items():
        if scores.get(metric) is not None:
            weighted_sum += scores[metric] * weight
            total_weight += weight

    global_01 = weighted_sum / total_weight if total_weight > 0 else 0.0
    return {
        "scores_individuels": scores,
        "ponderation": WEIGHTS,
        "score_global_100": round(global_01 * 100, 2),
    }

def interpret_score(score: float) -> str:
    if score >= 90: return "Excellent"
    if score >= 80: return "Très bon"
    if score >= 70: return "Bon"
    if score >= 60: return "Moyen"
    if score >= 50: return "Faible"
    return "Insuffisant"

def print_report(scores_info: Dict, n_samples: int):
    scores = scores_info["scores_individuels"]
    global_s = scores_info["score_global_100"]
    print("\n" + "=" * 60)
    print("   RAPPORT D'ÉVALUATION RAGAS")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\n   Questions évaluées : {n_samples:,}")
    print(f"\n   SCORE GLOBAL : {global_s:.2f} / 100")
    print(f"   Niveau       : {interpret_score(global_s)}")
    print("=" * 60)

def save_report(scores_info: Dict, n_samples: int, path: str):
    report = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "n_samples": n_samples,
            "model": CONFIG["vllm_model"],
        },
        "scores": scores_info,
        "interpretation": interpret_score(scores_info["score_global_100"]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Rapport sauvegardé : {path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "=" * 60)
    logger.info("   RAGAS EVAL ROBUST — ANSTAT-IA")
    logger.info(f"   Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    records = load_ragas_jsonl(CONFIG["input_jsonl"])
    accumulated = evaluate_in_batches(records)
    scores_info = compute_global_score(accumulated)
    print_report(scores_info, len(records))
    save_report(scores_info, len(records), CONFIG["output_report"])
    delete_checkpoint(CONFIG["checkpoint_file"])

    logger.info(f"🎉 Terminé ! Score global : {scores_info['score_global_100']:.2f}/100")

if __name__ == "__main__":
    main()