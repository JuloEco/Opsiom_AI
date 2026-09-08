# -*- coding: utf-8 -*-
"""
eval_prompts.py — Harnais d'évaluation qualitative pour Opsiom.

Rejoue un jeu de prompts FIXE (facile / moyen / narratif / instruction) à
travers le checkpoint actuel et sauvegarde les réponses dans un fichier JSON
horodaté, pour pouvoir comparer objectivement une run d'entraînement à
l'autre plutôt que de juger "à l'oreille" sur des prompts différents à
chaque fois.

Utilisation (à placer dans le même dossier que test_IA.py, sur ton Drive) :
    %run eval_prompts.py

Ça produit :
    eval_results/eval_2026-08-16_14-32-05.json   (résultats détaillés)
    eval_results/eval_history.md                  (tableau comparatif cumulé)

Chaque nouvelle exécution APPEND une colonne au tableau comparatif, sans
écraser les précédentes — pour voir l'évolution au fil des runs.
"""

import os
import sys
import json
import random
from datetime import datetime

import torch

# On importe directement les classes de test_IA.py (doit être dans le même
# dossier). Pas de duplication de la logique de génération/formatage — on
# veut évaluer EXACTEMENT ce que produit test_IA.py en usage réel, pas une
# réimplémentation qui pourrait diverger.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_IA import (  # noqa: E402
    FrenchTokenizerWrapper, ModelArgs, ModernLLM, StreamingInferenceEngine,
)

# ⚠️ Graine fixe : sans ça, deux évaluations du MÊME checkpoint donnent des
# réponses différentes à cause du tirage aléatoire de l'échantillonnage
# (temperature/top_k/top_p), rendant toute comparaison avant/après invalide.
# Constaté concrètement : deux runs sur le même chat_model.pt (même mtime)
# ont donné des qualités très différentes par simple chance de tirage.
EVAL_SEED = 1337

# ============================================================================
# Jeu de prompts fixe — NE PAS modifier d'une run à l'autre si tu veux une
# comparaison valide. Si tu veux tester d'autres prompts, ajoute-les dans une
# nouvelle catégorie plutôt que de changer les existants.
# ============================================================================

EVAL_PROMPTS = {
    "facile_factuel": [
        "Quelle est la capitale de la France ?",
        "Combien de continents y a-t-il ?",
        "Quelles sont les couleurs primaires ?",
        "Quel est le plus grand océan du monde ?",
    ],
    "definition_courte": [
        "Qu'est-ce que la photosynthèse ?",
        "Explique ce qu'est un volcan.",
        "C'est quoi un synonyme ?",
    ],
    "narratif_completion": [
        "Il était une fois un petit chat qui",
        "Dans la forêt, un lapin trouva",
        "Ce matin-là, la petite fille décida de",
    ],
    "narratif_instruction": [
        "Raconte-moi une histoire.",
        "Écris-moi un petit conte.",
    ],
    "instruction_courte": [
        "Donne-moi un synonyme de 'content'.",
        "Écris une phrase avec le mot 'jardin'.",
        "Traduis 'bonjour' en anglais.",
    ],
}

GEN_KWARGS = dict(max_new_tokens=80, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.3)


def load_model_and_tokenizer():
    """Reproduit la logique de chargement de test_IA.py::main() (chemins
    Drive puis local, détection auto de la config depuis le checkpoint)."""
    drive_dir = "/content/drive/MyDrive/mini_llm_fr"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_path = os.path.join(script_dir, "chat_model.pt")
    tokenizer_path = os.path.join(script_dir, "fr_bpe_tokenizer.json")

    if not os.path.exists(checkpoint_path) and os.path.exists(os.path.join(drive_dir, "chat_model.pt")):
        checkpoint_path = os.path.join(drive_dir, "chat_model.pt")
    if not os.path.exists(checkpoint_path) and os.path.exists(os.path.join(drive_dir, "best_model.pt")):
        checkpoint_path = os.path.join(drive_dir, "best_model.pt")

    if not os.path.exists(tokenizer_path) and os.path.exists(os.path.join(drive_dir, "fr_bpe_tokenizer.json")):
        tokenizer_path = os.path.join(drive_dir, "fr_bpe_tokenizer.json")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Aucun checkpoint trouvé (cherché: '{checkpoint_path}').")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer introuvable (cherché: '{tokenizer_path}').")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = FrenchTokenizerWrapper(tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        val_loss = checkpoint.get("val_loss", "N/A")
        step = checkpoint.get("step", "N/A")
        saved_args = checkpoint.get("args", None)
    else:
        state_dict = checkpoint
        val_loss, step, saved_args = "N/A", "N/A", None

    if saved_args is not None and hasattr(saved_args, "vocab_size"):
        args = saved_args
        args.device = device
    else:
        emb_shape = state_dict["tok_embeddings.weight"].shape
        vocab_size_ckpt, dim_ckpt = emb_shape[0], emb_shape[1]
        layer_indices = {int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")}
        n_layers_ckpt = len(layer_indices) if layer_indices else 8
        wk_shape = state_dict["layers.0.attn.wk.weight"].shape
        n_heads_ckpt = 8
        head_dim = dim_ckpt // n_heads_ckpt
        n_kv_heads_ckpt = wk_shape[0] // head_dim
        args = ModelArgs(
            vocab_size=vocab_size_ckpt, dim=dim_ckpt, n_layers=n_layers_ckpt,
            n_heads=n_heads_ckpt, n_kv_heads=n_kv_heads_ckpt, device=device,
        )

    model = ModernLLM(args).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    meta = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_mtime": datetime.fromtimestamp(os.path.getmtime(checkpoint_path)).isoformat(timespec="seconds"),
        "step": step,
        "val_loss": val_loss,
        "dim": args.dim,
        "n_layers": args.n_layers,
    }
    return model, tokenizer, meta


def run_eval():
    print("🔍 Chargement du modèle et du tokenizer...")
    model, tokenizer, meta = load_model_and_tokenizer()
    print(f"✅ Checkpoint: {meta['checkpoint_path']} (step={meta['step']}, val_loss={meta['val_loss']}, "
          f"dim={meta['dim']}, layers={meta['n_layers']})")

    engine = StreamingInferenceEngine(model, tokenizer)

    results = {"meta": {**meta, "eval_seed": EVAL_SEED}, "timestamp": datetime.now().isoformat(timespec="seconds"), "categories": {}}

    # Graine fixée juste avant la génération (pas avant le chargement du
    # modèle, pour ne pas dépendre de l'ordre des opérations de chargement).
    torch.manual_seed(EVAL_SEED)
    random.seed(EVAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EVAL_SEED)

    for category, prompts in EVAL_PROMPTS.items():
        print(f"\n--- {category} ---")
        results["categories"][category] = []
        for prompt in prompts:
            # Un seul tour, sans historique préalable — chaque prompt est
            # indépendant pour ne pas mélanger l'effet du prompt lui-même
            # avec des artefacts multi-tours.
            for _ in engine.stream_generate_from_history([{"role": "user", "content": prompt}], **GEN_KWARGS):
                pass
            response = engine.last_response.strip()
            print(f"👉 {prompt}\n🤖 {response}\n")
            results["categories"][category].append({"prompt": prompt, "response": response})

    return results


def save_results(results):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "eval_results")
    os.makedirs(out_dir, exist_ok=True)

    # Garde-fou : prévenir si ce checkpoint a déjà été évalué (mtime identique
    # à une run précédente) — signe qu'aucun (ré-)entraînement n'a eu lieu
    # entre les deux évaluations, donc que toute différence de sortie n'est
    # QUE de la variance d'échantillonnage, pas un vrai progrès du modèle.
    previous_jsons = sorted(
        f for f in os.listdir(out_dir) if f.startswith("eval_") and f.endswith(".json")
    )
    if previous_jsons:
        with open(os.path.join(out_dir, previous_jsons[-1]), "r", encoding="utf-8") as f:
            last = json.load(f)
        if last.get("meta", {}).get("checkpoint_mtime") == results["meta"]["checkpoint_mtime"]:
            print(
                f"⚠️  ATTENTION : même checkpoint_mtime que la dernière évaluation "
                f"('{previous_jsons[-1]}') — aucun entraînement n'a eu lieu entre les "
                f"deux runs. Les différences de sortie, s'il y en a, ne reflètent que "
                f"la variance d'échantillonnage (même graine ici, donc ça ne devrait "
                f"même plus arriver), pas un vrai changement du modèle."
            )

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = os.path.join(out_dir, f"eval_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Résultats détaillés sauvegardés: '{json_path}'")

    # Tableau comparatif cumulé, en append — une section par run.
    md_path = os.path.join(out_dir, "eval_history.md")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {results['timestamp']} — {results['meta']['checkpoint_path']}\n")
        f.write(f"step={results['meta']['step']}, val_loss={results['meta']['val_loss']}, "
                f"dim={results['meta']['dim']}, layers={results['meta']['n_layers']}\n\n")
        for category, items in results["categories"].items():
            f.write(f"**{category}**\n\n")
            for item in items:
                f.write(f"- *{item['prompt']}*\n  → {item['response']}\n")
            f.write("\n")
    print(f"📝 Historique comparatif mis à jour: '{md_path}'")


if __name__ == "__main__":
    results = run_eval()
    save_results(results)
