# ============================================================================
# Mini-LLM style LLaMA/Qwen — script complet prêt pour une cellule Colab
# RMSNorm + RoPE + SwiGLU + Attention SDPA (FlashAttention) + KV-Cache
# ============================================================================

# --- Dépendances (décommente si tu es sur Colab, sinon `pip install` en local) ---
# %pip install -q datasets tokenizers torch accelerate

import os

# ⚠️ Doit être positionné AVANT tout import de torch qui toucherait CUDA:
# suggéré directement par le message d'erreur OOM rencontré ("If reserved but
# unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True").
# Réduit la fragmentation mémoire du cache d'allocateur CUDA de PyTorch — utile
# vu qu'on est déjà à la limite de VRAM du T4 avec Opsiom-Large.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ⚠️ Sans ça, CHAQUE upload_file()/hf_hub_download() du système de sauvegarde
# externe (voir push_backup_to_hf / try_restore_backup_from_hf plus bas)
# affiche une barre de progression tqdm complète dans la sortie du notebook.
# Avec un push toutes les quelques dizaines de steps en tout début
# d'entraînement (val_loss qui s'améliore souvent) + un push "latest" toutes
# les 10 minutes, ça pollue rapidement la sortie de plusieurs milliers de
# lignes, ralentit l'affichage Kaggle/Colab, et gonfle inutilement la taille
# du commit final. On garde les print() explicites de push_backup_to_hf
# (concis, un par sauvegarde) comme seule trace utile.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import re
import json
import time
import sys
import math
import inspect
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from accelerate import Accelerator
from dataclasses import dataclass

# ⚠️ Ampere (RTX A5000, compute capability 8.6) supporte le TF32 pour les
# matmuls/convs qui restent en fp32 pendant l'entraînement (ex: certaines
# réductions internes non couvertes par l'autocast bf16) — gain de vitesse
# quasi gratuit, sans perte de précision significative pour ce genre
# d'entraînement. cudnn.benchmark=True est sûr ici car block_size (donc la
# forme des tenseurs d'entrée) est constant tout au long du run: cuDNN peut
# mettre en cache le meilleur algorithme trouvé au 1er step au lieu de le
# redécouvrir. Sur un T4 (Turing, pas Ampere) le gain TF32 est nul mais sans
# risque non plus — ces lignes sont sûres sur toute génération de GPU NVIDIA.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ----------------------------------------------------------------------------
# Sortie non-bufferisée: dans un notebook Kaggle/Colab, stdout n'est pas un
# vrai terminal (c'est un pipe, ou même un objet `OutStream` custom d'ipykernel
# qui n'a pas de méthode `.reconfigure()`), donc Python bufferise entièrement
# au lieu d'écrire ligne par ligne. Résultat: les print() n'apparaissent qu'au
# bout d'un moment, voire seulement quand le processus se termine (ou est
# interrompu). On force ici `flush=True` sur CHAQUE print() via un monkeypatch
# — ça fonctionne quel que soit le type de flux (contrairement à
# `sys.stdout.reconfigure()`, absent sur l'OutStream de Kaggle), aussi bien
# dans le processus principal que dans chaque processus enfant spawné (ce
# bloc, hors de `if __name__ == "__main__"`, s'exécute dans les deux cas).
import builtins as _builtins

# ⚠️ Garde-fou anti-double-patch: si ce script s'exécute une 2e fois dans le
# MÊME kernel (ex: on relance la cellule après un crash, sans redémarrer),
# `_builtins.print` est déjà le `_flushing_print` du 1er passage. Sans ce
# garde-fou, `_original_print` capturerait cette version déjà patchée plutôt
# que le vrai print d'origine, et `_flushing_print` finirait par s'appeler
# lui-même indéfiniment (RecursionError observée en pratique) dès que le
# namespace du module est réutilisé d'une exécution à l'autre. Le marqueur
# `_is_flushing_wrapper` permet de détecter ce cas et de ne PAS re-patcher.
if not getattr(_builtins.print, "_is_flushing_wrapper", False):
    _original_print = _builtins.print

    def _flushing_print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        _original_print(*args, **kwargs)

    _flushing_print._is_flushing_wrapper = True
    _builtins.print = _flushing_print


# ============================================================================
# Détection d'environnement — Colab (Drive), Kaggle, ou local
# ============================================================================
# Sur Colab, monte le Drive et fait pointer TOKENIZER_CACHE_PATH / CHECKPOINT_PATH
# / PRETRAIN_BIN_PATH dedans, pour que tokenizer, checkpoints et corpus survivent
# à la fermeture de la session. Sur Kaggle, pas d'équivalent Drive automatique:
# on utilise /kaggle/working (le dossier "Output" du kernel). ⚠️ Contrairement à
# Drive, /kaggle/working n'est PAS automatiquement persistant d'une session à
# l'autre en édition interactive — il ne survit que si tu fais "Save Version"
# (commit), et pour vraiment reprendre entre plusieurs sessions il faut ensuite
# ajouter cet Output comme Input Dataset de ta prochaine session. Hors Colab et
# hors Kaggle (exécution locale), on retombe sur le répertoire courant.

DRIVE_SAVE_DIR = "/content/drive/MyDrive/mini_llm_fr"
KAGGLE_SAVE_DIR = "/kaggle/working/mini_llm_fr"


# ----------------------------------------------------------------------------
# 🔑 Authentification Hugging Face sans conflit d'environnement
# ----------------------------------------------------------------------------
# ⚠️ Ce bloc est du code de niveau module, hors de `if __name__ == "__main__"`:
# avec torch.multiprocessing.spawn (méthode 'spawn'), CHAQUE processus enfant
# réimporte et réexécute intégralement ce script — donc ce bloc tournerait une
# fois par GPU en plus du processus parent. `huggingface_hub.login()` pose un
# verrou fichier (filelock) sur le cache HF pour écrire le token: plusieurs
# processus qui l'appellent en même temps peuvent entrer en contention sur ce
# verrou et rester bloqués indéfiniment (c'est très probablement ce qui a causé
# le blocage observé juste après le lancement des processus). On ne fait donc
# l'appel réseau `login()` que dans le tout premier processus (le parent);
# les processus spawnés héritent de toute façon de `HF_TOKEN` via
# l'environnement (les variables d'env sont copiées au processus enfant), donc
# il leur suffit de le relire sans refaire l'appel réseau/verrou.
_IS_SPAWNED_CHILD = mp.current_process().name != "MainProcess"

hf_token = os.environ.get("HF_TOKEN")

# Si non présent dans l'OS, on cherche dans les secrets Kaggle
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        hf_token = None

if hf_token:
    hf_token = hf_token.strip()
    if _IS_SPAWNED_CHILD:
        # Processus enfant (spawn): pas de login() réseau, juste s'assurer que
        # la variable d'environnement est bien présente pour ce processus.
        os.environ["HF_TOKEN"] = hf_token
    else:
        try:
            from huggingface_hub import login
            # On passe le token directement à login()
            login(token=hf_token, add_to_git_credential=False)
            # On met à jour l'environnement APRÈS la validation réussie
            os.environ["HF_TOKEN"] = hf_token
            print("🔑 Authentification Hugging Face réussie !")
        except Exception as e:
            print(f"⚠️ Échec de l'authentification : {e}")

def _is_kaggle() -> bool:
    return os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def _detect_env_and_get_save_dir() -> str:
    """Détecte l'environnement d'exécution et renvoie le dossier de sauvegarde à
    utiliser pour le tokenizer, les checkpoints et le corpus de pré-entraînement."""
    try:
        from google.colab import drive  # disponible uniquement sur Colab
        print("📎 Montage de Google Drive...")
        drive.mount("/content/drive")
        os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
        print(f"✅ Google Drive monté — sauvegardes dans {DRIVE_SAVE_DIR}")
        return DRIVE_SAVE_DIR
    except:
        pass

    if _is_kaggle():
        os.makedirs(KAGGLE_SAVE_DIR, exist_ok=True)
        print(f"ℹ️ Environnement Kaggle détecté — sauvegardes dans {KAGGLE_SAVE_DIR}.")
        print("   ↳ Pense à faire 'Save Version' (commit) pour conserver ce dossier, "
              "puis à le réimporter comme Input Dataset la prochaine fois pour reprendre.")
        return KAGGLE_SAVE_DIR

    print("ℹ️ Ni Colab ni Kaggle détecté — sauvegarde en local dans le répertoire courant.")
    return "."


def _restore_kaggle_dataset_cache(save_dir: str) -> None:
    """Si une session Kaggle précédente a été sauvegardée (bouton 'Save
    Version') puis réimportée comme Input Dataset de cette nouvelle session,
    restaure automatiquement le tokenizer / corpus de pré-entraînement /
    checkpoint déjà construits — pour éviter de refaire les ~2h30 de
    téléchargement + tokenisation à chaque nouvelle session Kaggle (rappel:
    /kaggle/working est vidé entre deux sessions, contrairement à Drive sur
    Colab). Ne copie que les fichiers absents localement: n'écrase jamais un
    fichier déjà présent dans `save_dir` (au cas où la session en cours ait
    déjà progressé plus loin que l'ancien commit importé)."""
    if not _is_kaggle():
        return
    import glob
    import shutil

    candidates = glob.glob("/kaggle/input/*/mini_llm_fr") + glob.glob("/kaggle/input/*/*/mini_llm_fr")
    if not candidates:
        return
    src_dir = candidates[0]
    restored = []
    for fname in ("pretrain_corpus.bin", "pretrain_corpus_meta.json", "fr_bpe_tokenizer.json", "best_model.pt"):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(save_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            restored.append(fname)
    if restored:
        print(f"♻️  Fichiers restaurés depuis le dataset Kaggle importé ({src_dir}): {', '.join(restored)}")
    else:
        print(f"ℹ️ Dataset Kaggle importé détecté ({src_dir}) mais rien à restaurer "
              f"(fichiers déjà présents localement ou absents du dataset).")


_SAVE_DIR = _detect_env_and_get_save_dir()
_restore_kaggle_dataset_cache(_SAVE_DIR)

# --- Multi-GPU (utile sur Kaggle: accélérateur "GPU T4 x2") ---
# WORLD_SIZE > 1 déclenche un entraînement distribué (DistributedDataParallel):
# un processus par GPU, un identique modèle sur chacun, gradients synchronisés
# à chaque step. Voir main_worker() et le lancement dans `if __name__ == "__main__"`.
# ⚠️ BUG CORRIGÉ (était: FORCE_SINGLE_GPU = True en dur, donc WORLD_SIZE=1 même
# avec 2 GPU physiquement disponibles sur Kaggle T4x2). La cause du problème
# historique (logs dupliqués, signe que chaque GPU s'entraînait indépendamment
# sans réel échange de gradients) n'était PAS le double GPU en lui-même, mais
# le fait qu'Accelerator() ne détectait pas correctement le contexte distribué
# quand les processus sont lancés via torch.multiprocessing.spawn (au lieu de
# `accelerate launch`/`notebook_launcher`). Ça a été corrigé dans main_worker()
# en initialisant explicitement le process group PyTorch
# (torch.distributed.init_process_group) AVANT de construire l'Accelerator —
# voir plus bas. Le double GPU peut donc être réactivé en toute sécurité.
FORCE_SINGLE_GPU = False
WORLD_SIZE = 1 if FORCE_SINGLE_GPU else (torch.cuda.device_count() if torch.cuda.is_available() else 1)


# ============================================================================
# 🛟 Sauvegarde externe (Hugging Face Hub) — survit à un crash/coupure Kaggle
# ============================================================================
# Le vrai problème avec /kaggle/working: il n'est PAS persistant tant qu'un
# "Save Version" (commit) n'a pas abouti. Un crash, un OOM-kill, une coupure
# réseau ou électrique AVANT ce commit final efface tout, même si
# best_model.pt vient d'être écrit sur disque une seconde plus tôt. On pousse
# donc chaque nouveau meilleur checkpoint (+ le tokenizer) vers un repo HF Hub
# PRIVÉ dès qu'il est sauvegardé localement: le fichier quitte Kaggle
# immédiatement, indépendamment de ce qui arrive à la session ensuite.
#
# Prérequis: HF_TOKEN doit avoir les droits d'écriture (un token "write", pas
# "read"), et HF_BACKUP_REPO_ID doit pointer vers un repo que ce token peut
# créer/modifier (il est créé automatiquement s'il n'existe pas encore).
HF_BACKUP_ENABLED = True
HF_BACKUP_REPO_ID = "JuloEco/opsiom-fr-checkpoints"  # ⚠️ adapte à ton propre namespace HF
HF_BACKUP_PRIVATE = True
# Filet de sécurité supplémentaire: même sans nouvelle amélioration de
# val_loss (donc sans nouveau "best_model.pt"), on repousse l'état ACTUEL du
# modèle toutes les N secondes, sous un nom distinct ("latest_model.pt"). Sans
# ça, une longue période sans amélioration + un crash = retour au dernier best
# parfois vieux de plusieurs heures, alors qu'un état plus récent (même non
# "meilleur") existait juste avant le crash.
HF_BACKUP_LATEST_INTERVAL_SECONDS = 600  # 10 min

_hf_backup_api = None


def _get_hf_backup_api():
    """Instancie paresseusement le client HfApi (une seule fois), et
    crée le repo de sauvegarde s'il n'existe pas déjà."""
    global _hf_backup_api
    if _hf_backup_api is not None:
        return _hf_backup_api
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.create_repo(repo_id=HF_BACKUP_REPO_ID, private=HF_BACKUP_PRIVATE, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"⚠️ Impossible de créer/vérifier le repo de sauvegarde HF '{HF_BACKUP_REPO_ID}' ({e}). "
              f"La sauvegarde externe est désactivée pour cette session.")
        return None
    _hf_backup_api = api
    return api


def push_backup_to_hf(local_path: str, path_in_repo: str) -> None:
    """Pousse un fichier vers le repo de sauvegarde HF Hub. Ne lève JAMAIS
    d'exception: un problème réseau ponctuel ne doit pas interrompre
    l'entraînement — on log juste un avertissement et on continue."""
    if not HF_BACKUP_ENABLED or not os.path.exists(local_path):
        return
    api = _get_hf_backup_api()
    if api is None:
        return
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=HF_BACKUP_REPO_ID,
            repo_type="model",
        )
        size_mb = os.path.getsize(local_path) / 1e6
        print(f"   ↳ 🛟 Sauvegarde externe: '{path_in_repo}' poussé vers {HF_BACKUP_REPO_ID} ({size_mb:.1f} Mo)")
    except Exception as e:
        print(f"   ↳ ⚠️ Échec de la sauvegarde externe de '{path_in_repo}' ({e}) — entraînement non interrompu.")


def try_restore_backup_from_hf(local_path: str, path_in_repo: str) -> bool:
    """Tente de restaurer un fichier depuis le repo de sauvegarde HF Hub si le
    fichier local est absent — utile après une session Kaggle perdue sans
    'Save Version', pour reprendre automatiquement sans réimport manuel.
    Retourne True si un fichier a été restauré."""
    if not HF_BACKUP_ENABLED or os.path.exists(local_path):
        return False
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=HF_BACKUP_REPO_ID, filename=path_in_repo, repo_type="model")
        import shutil
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        shutil.copy2(downloaded, local_path)
        print(f"♻️  Restauré depuis la sauvegarde externe HF Hub: '{path_in_repo}' -> '{local_path}'")
        return True
    except Exception as e:
        print(f"ℹ️ Pas de sauvegarde externe utilisable pour '{path_in_repo}' ({e}).")
        return False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelArgs:
    """Configuration hyperparamétrique pour architecture Transformer type LLaMA/Qwen."""
    vocab_size: int = 16000   # Valeur par défaut — écrasée dynamiquement par la taille
                               # réelle du vocabulaire du tokenizer français entraîné plus bas
    dim: int = 1024           # Dimension des embeddings — Opsiom-Large
    n_layers: int = 16        # Nombre de blocs Transformer
    n_heads: int = 16         # Nombre de têtes d'attention (Query)
    n_kv_heads: int | None = 4  # Grouped-Query Attention (GQA): 4 têtes K/V pour 16 têtes Q
    max_seq_len: int = 512    # Fenêtre de contexte
    dropout: float = 0.1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Hyperparamètres d'entraînement — modifie-les librement
N_STORIES = 20000          # Plafond d'histoires TinyStories-French utilisées (le dataset n'en
                            # contient qu'environ 1000 au total, donc en pratique tout est utilisé)
TOKENIZER_VOCAB_SIZE = 16000   # Taille cible du vocabulaire du tokenizer BPE français
TOKENIZER_CACHE_PATH = os.path.join(_SAVE_DIR, "fr_bpe_tokenizer.json")
WIKI_CONFIG = "wikitext-72"    # Plus grande des deux configs d'asi/wikitext_fr (quality + good articles)
WIKI_MAX_CHARS = 25_000_000    # Plafond de caractères Wikipedia chargés (tokenizer + corpus LM)
# MAX_STEPS: 1000 steps à batch=32/block=256 ne couvre qu'~1 epoch sur le corpus
# (TinyStories-FR + Wikipedia ≈ 5-6M tokens). Pour un modèle de ~26M paramètres,
# c'est trop peu pour stabiliser les statistiques de sous-mots (cause principale
# des artefacts type "dçant", "garès êtreux"). On vise ~15-20 epochs sur les
# données disponibles plutôt qu'un budget de compute abstrait — ajustez à la
# hausse si votre val_loss continue de baisser à la fin de l'entraînement.
MAX_STEPS = 8000            # ~8-10 epochs sur le corpus combiné (était 1000)
WARMUP_STEPS = 400           # gardé à 5% de MAX_STEPS, comme avant
# ⚠️ BATCH_SIZE remonté 8 -> 24 pour la RTX A5000 (24 Go VRAM, contre 14,56 Go
# utilisables sur le T4 qui avait motivé la valeur de 8). C'est un point de
# départ raisonnable pour Opsiom-Large (196,77M params, max_seq_len=512, bf16
# natif sur Ampere) — surveille `nvidia-smi` pendant les premiers steps: s'il
# reste beaucoup de VRAM libre, remonte encore (32, 40...) par paliers ; en
# cas d'OOM, redescends. Le TF32/bf16 + le batch plus gros font l'essentiel du
# gain de throughput par rapport au réglage T4.
BATCH_SIZE = 24
# GRAD_ACCUM_STEPS abaissé en conséquence: avec un batch micro déjà bien plus
# gros (24 au lieu de 8), moins d'accumulation suffit pour un batch EFFECTIF
# confortable (24 x 2 = 48 en mono-GPU, ou 24 x n_GPU x 2 en multi-GPU) tout
# en gardant plus de vraies mises à jour de poids par unité de temps (moins de
# micro-steps "gaspillés" avant chaque step réel) — l'A5000 a largement la
# VRAM pour se permettre moins d'accumulation. Ajuste à la hausse si tu veux
# un batch effectif plus grand sans remonter BATCH_SIZE.
GRAD_ACCUM_STEPS = 4 if FORCE_SINGLE_GPU else 2
MAX_LR = 3e-4
MIN_LR = 3e-5
EVAL_INTERVAL = 200        # Évaluation + sauvegarde du meilleur modèle tous les N steps
GEN_INTERVAL = 400         # Génération d'un échantillon de contrôle tous les N steps
CHECKPOINT_PATH = os.path.join(_SAVE_DIR, "best_model.pt")
LATEST_CHECKPOINT_PATH = os.path.join(_SAVE_DIR, "latest_model.pt")
SEED = 1337

# Cible de tokens pour le pré-entraînement à grande échelle (modèle "Large").
# Règle Chinchilla (~20 tokens / paramètre): 196.77M params x 20 ≈ 3.9355e9.
# Ajuste ce chiffre au nombre de paramètres réel de ton modèle si tu changes
# ModelArgs (dim/n_layers/n_heads) — ce fichier ne fixe QUE la taille du corpus,
# pas l'architecture.
TARGET_PRETRAIN_TOKENS = 3_935_400_000
PRETRAIN_BIN_PATH = os.path.join(_SAVE_DIR, "pretrain_corpus.bin")
PRETRAIN_META_PATH = os.path.join(_SAVE_DIR, "pretrain_corpus_meta.json")
# Nombre de tokens réservés à la validation, plafonné: à 3,9 milliards de tokens,
# 5% donnerait ~195M tokens de val — bien plus que nécessaire pour une estimation
# stable de la val_loss, et ça gaspillerait du budget de tokens d'entraînement
# chèrement acquis (téléchargement + tokenisation).
VAL_TOKENS_CAP = 20_000_000

# Reprend l'entraînement depuis CHECKPOINT_PATH s'il existe, au lieu de
# repartir de poids aléatoires. Pratique pour étendre un entraînement déjà
# fait (ex: vous aviez tourné 1000 steps, vous voulez continuer) sans perdre
# ce qui a déjà été appris. Le tokenizer/vocab doit être identique.
RESUME_FROM_CHECKPOINT = True


# ============================================================================
# Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (plus rapide et stable que LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # (dim,) gain appris

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (B, T, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calcul en float32 pour la stabilité numérique, recast dans le dtype d'origine
        out = self._norm(x.float()).type_as(x)  # (B, T, C)
        return out * self.weight  # (B, T, C) * (C,) broadcast


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) — précalcule cos/sin pour toutes les positions."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "head_dim doit être pair pour RoPE"
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len).float()  # (max_seq_len,)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # (max_seq_len, dim)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)  # (max_seq_len, dim)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0):
        # Positions [start_pos, start_pos + seq_len) — essentiel pour le décodage avec KV-Cache
        cos = self.cos_cached[start_pos:start_pos + seq_len].to(dtype=x.dtype, device=x.device)  # (T, head_dim)
        sin = self.sin_cached[start_pos:start_pos + seq_len].to(dtype=x.dtype, device=x.device)  # (T, head_dim)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Fait pivoter la moitié des dimensions: [-x2, x1] où x = [x1, x2]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applique la rotation RoPE sur les Query et Key tensors.

    xq, xk: (B, n_heads, T, head_dim) ; cos, sin: (T, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    xq_rotated = (xq * cos) + (rotate_half(xq) * sin)  # (B, n_heads, T, head_dim)
    xk_rotated = (xk * cos) + (rotate_half(xk) * sin)  # (B, n_kv_heads, T, head_dim)
    return xq_rotated, xk_rotated


# ============================================================================
# SwiGLU FeedForward
# ============================================================================

class SwiGLUFeedForward(nn.Module):
    """Couche MLP SwiGLU (Gated Linear Unit avec SiLU) utilisée dans LLaMA/Qwen."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(8 * args.dim / 3)
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)  # down
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)  # up
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))     # (B, T, hidden_dim)
        up = self.w3(x)               # (B, T, hidden_dim)
        out = self.w2(gate * up)      # (B, T, dim)
        return self.dropout(out)


# ============================================================================
# Attention (SDPA / FlashAttention) avec support KV-Cache
# ============================================================================

class ModernCausalAttention(nn.Module):
    """Attention causale moderne avec PyTorch SDPA (FlashAttention / Scaled Dot-Product)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        assert args.n_heads % self.n_kv_heads == 0, "n_heads doit être divisible par n_kv_heads (GQA)"
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dropout_p = args.dropout

        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache=None):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # (B, n_heads, T, head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)

        q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat((past_k, k), dim=2)  # (B, n_kv_heads, T_past+T, head_dim)
                v = torch.cat((past_v, v), dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # (B, n_heads, T_kv, head_dim)
            v = v.repeat_interleave(self.n_rep, dim=1)

        is_causal = kv_cache is None or k.shape[2] == q.shape[2]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )  # (B, n_heads, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)  # (B, T, dim)
        y = self.resid_dropout(self.wo(y))

        if kv_cache is not None:
            return y, new_kv_cache
        return y


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = ModernCausalAttention(args)
        self.ffn = SwiGLUFeedForward(args)
        self.norm1 = RMSNorm(args.dim, eps=args.norm_eps)
        self.norm2 = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache=None):
        if kv_cache is not None:
            attn_out, new_kv_cache = self.attn(self.norm1(x), cos, sin, kv_cache=kv_cache)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, new_kv_cache
        else:
            x = x + self.attn(self.norm1(x), cos, sin)
            x = x + self.ffn(self.norm2(x))
            return x


# ============================================================================
# Modèle complet
# ============================================================================

class ModernLLM(nn.Module):
    """Architecture complète GPT/LLaMA auto-régressive."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)

        head_dim = args.dim // args.n_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=args.max_seq_len, theta=args.rope_theta)

        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm_f = RMSNorm(args.dim, eps=args.norm_eps)

        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.lm_head.weight  # Weight Tying

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("w2.weight") or name.endswith("wo.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # Les poids liés (tok_embeddings == lm_head) ne sont comptés qu'une fois:
        # on déduplique par id() du tenseur avant de sommer.
        unique_params = {id(p): p for p in self.parameters()}
        return sum(p.numel() for p in unique_params.values())

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list | None = None,
        start_pos: int = 0,
    ):
        B, T = tokens.shape
        assert start_pos + T <= self.args.max_seq_len, (
            f"Position {start_pos + T} > max_seq_len {self.args.max_seq_len}"
        )

        x = self.tok_embeddings(tokens)  # (B, T, dim)
        x = self.dropout(x)

        cos, sin = self.rope(x, seq_len=T, start_pos=start_pos)

        if kv_caches is not None:
            new_kv_caches = []
            for i, layer in enumerate(self.layers):
                x, layer_cache = layer(x, cos, sin, kv_cache=kv_caches[i])
                new_kv_caches.append(layer_cache)
        else:
            for layer in self.layers:
                x = layer(x, cos, sin)
            new_kv_caches = None

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        if kv_caches is not None:
            return logits, loss, new_kv_caches
        return logits, loss

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        generated_ids: torch.Tensor | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Échantillonne le prochain token. logits: (1, vocab_size)."""

        # --- Pénalité de répétition (style HuggingFace) ---
        # Pour chaque token déjà généré, on divise son logit par la pénalité s'il est
        # positif (on le rend moins probable) ou on le multiplie s'il est négatif
        # (même effet: on pousse le score vers -inf). Casse les boucles de mots répétés.
        if repetition_penalty != 1.0 and generated_ids is not None and generated_ids.numel() > 0:
            unique_ids = torch.unique(generated_ids)
            prev_logits = logits[0, unique_ids]  # (n_unique,)
            penalized = torch.where(
                prev_logits > 0,
                prev_logits / repetition_penalty,
                prev_logits * repetition_penalty,
            )
            logits[0, unique_ids] = penalized

        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)

        logits = logits / temperature

        if top_k is not None:
            top_k_clamped = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k_clamped)
            threshold = v[:, [-1]]
            logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(logits, dim=-1)

        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

        return torch.multinomial(probs, num_samples=1)  # (1, 1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer: "FrenchTokenizerWrapper",
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int | None = 40,
        repetition_penalty: float = 1.3,
    ):
        """Génération auto-régressive avec KV-Cache, temperature, top-k, top-p et
        pénalité de répétition. Le prompt est traité en une seule passe (prefill),
        puis chaque nouveau token n'est calculé qu'une seule fois (decode step)."""
        self.eval()
        device = next(self.parameters()).device

        token_ids = tokenizer.encode(prompt, allowed_special="all")
        tokens = torch.tensor([token_ids], dtype=torch.long, device=device)  # (1, T0)

        max_prompt_len = self.args.max_seq_len - 1
        tokens = tokens[:, -max_prompt_len:]

        kv_caches = [(None, None) for _ in range(self.args.n_layers)]

        logits, _, kv_caches = self.forward(tokens, kv_caches=kv_caches, start_pos=0)
        logits = logits[:, -1, :]
        cur_pos = tokens.shape[1]

        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(
                logits, temperature, top_k, top_p,
                generated_ids=tokens, repetition_penalty=repetition_penalty,
            )
            tokens = torch.cat((tokens, next_token), dim=1)

            if next_token.item() == tokenizer.eot_token:
                break
            if cur_pos >= self.args.max_seq_len:
                break

            logits, _, kv_caches = self.forward(next_token, kv_caches=kv_caches, start_pos=cur_pos)
            logits = logits[:, -1, :]
            cur_pos += 1

        self.train()
        return tokenizer.decode(tokens[0].tolist())


# ============================================================================
# Dataset
# ============================================================================

class TextDataset(torch.utils.data.Dataset):
    """Découpe un long corpus tokenisé en fenêtres (input, target) décalées d'un token."""

    def __init__(self, token_ids: list[int], block_size: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y


class TextDatasetMemmap(torch.utils.data.Dataset):
    """Équivalent de TextDataset, mais lit les tokens depuis un fichier binaire sur
    disque via np.memmap au lieu de tout garder en RAM. Indispensable au-delà de
    quelques centaines de millions de tokens: un tenseur de 3,9 milliards de
    tokens en int64 pèserait à lui seul ~31 Go, hors de portée d'une VM Colab."""

    def __init__(self, bin_path: str, dtype, start: int, end: int, block_size: int):
        self.bin_path = bin_path
        self.dtype = dtype
        self.start = start
        self.end = end  # exclusif
        self.block_size = block_size
        self._mm = None  # ouvert paresseusement, voir _ensure_mmap

    def _ensure_mmap(self):
        # np.memmap ne se transmet pas proprement aux processus worker d'un
        # DataLoader multi-process — on l'ouvre à la demande dans chaque worker
        # (le premier __getitem__ appelé dans ce processus l'initialise).
        if self._mm is None:
            self._mm = np.memmap(self.bin_path, dtype=self.dtype, mode="r")

    def __len__(self):
        return max(0, (self.end - self.start) - self.block_size)

    def __getitem__(self, idx: int):
        self._ensure_mmap()
        i = self.start + idx
        x = torch.from_numpy(self._mm[i: i + self.block_size].astype(np.int64))
        y = torch.from_numpy(self._mm[i + 1: i + 1 + self.block_size].astype(np.int64))
        return x, y


class RandomWindowIterableDataset(torch.utils.data.IterableDataset):
    """Dataset infini pour un corpus memmap gigantesque (potentiellement des
    milliards de fenêtres): pioche des positions de départ aléatoires une par
    une (`random.randint`), au lieu de matérialiser une permutation complète
    des indices comme le font `shuffle=True` / `DistributedSampler` par défaut.

    ⚠️ `DistributedSampler(..., shuffle=True)` et le `shuffle=True` standard
    d'un DataLoader appellent en interne `torch.randperm(len(dataset))` pour
    mélanger les indices. Sur un dataset de ~3,9 milliards de fenêtres, ça
    alloue un tenseur de ~31 Go rien que pour les indices (int64), PUIS le
    convertit en liste Python de 3,9 milliards d'objets `int` (des dizaines de
    Go supplémentaires, chaque int Python pesant bien plus que 8 octets). Sur
    Kaggle, cette tentative d'allocation dépasse la RAM système disponible et
    le noyau Linux tue le processus (SIGKILL) — c'est précisément ce qui s'est
    produit au tout premier step d'entraînement.

    Un tirage aléatoire AVEC remise (bootstrap), fenêtre par fenêtre, est
    statistiquement équivalent à un vrai mélange pour du pré-entraînement à
    cette échelle (on ne complète de toute façon jamais une epoch entière sur
    un corpus de cette taille en quelques sessions Kaggle), et ne matérialise
    jamais plus d'un seul index à la fois."""

    def __init__(self, bin_path: str, dtype, start: int, end: int, block_size: int, seed: int = 0):
        super().__init__()
        self.bin_path = bin_path
        self.dtype = dtype
        self.start = start
        self.end = end  # exclusif
        self.block_size = block_size
        self.seed = seed
        self._mm = None

    def _ensure_mmap(self):
        if self._mm is None:
            self._mm = np.memmap(self.bin_path, dtype=self.dtype, mode="r")

    def __iter__(self):
        self._ensure_mmap()
        # Graine différente par worker DataLoader (si num_workers > 0 un jour)
        # pour ne pas tirer exactement la même séquence dans chaque worker.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + worker_id)
        hi = self.end - self.block_size - 1
        while True:
            i = rng.randint(self.start, hi)
            x = torch.from_numpy(self._mm[i: i + self.block_size].astype(np.int64))
            y = torch.from_numpy(self._mm[i + 1: i + 1 + self.block_size].astype(np.int64))
            yield x, y


# ============================================================================
# Tokenizer BPE français — vocabulaire entraîné sur asi/wikitext_fr (Hugging Face)
# ============================================================================

class FrenchTokenizerWrapper:
    """Adapte un `tokenizers.Tokenizer` (BPE byte-level) à l'interface utilisée
    dans le reste du script: `.encode()`, `.decode()`, `.eot_token`."""

    def __init__(self, tokenizer):
        self._tok = tokenizer
        eot_id = tokenizer.token_to_id("<|endoftext|>")
        assert eot_id is not None, "Le tokenizer doit contenir le token spécial <|endoftext|>"
        self.eot_token = eot_id
        self.vocab_size = tokenizer.get_vocab_size()

    def encode(self, text: str, allowed_special: str = "all") -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def _import_datasets_module():
    try:
        return __import__("datasets")
    except ImportError as e:
        raise ImportError(
            "Le package 'datasets' n'est pas installé. Installez-le avec `pip install datasets`."
        ) from e


# ============================================================================
# Filtre anti-LaTeX (corpus Wikipedia)
# ============================================================================
_LATEX_COMMAND_RE = re.compile(r"\\(?:[a-zA-Z]+|[^a-zA-Z\s])")   # \frac, \alpha, \{, \\, ...
_LATEX_SCRIPT_RE = re.compile(r"[_^]\{[^{}]{0,80}\}")            # T_{ij}, x^{2}
_LATEX_INLINE_MATH_RE = re.compile(r"\${1,2}[^$\n]{1,200}\${1,2}")  # $...$ ou $$...$$
_LATEX_ENV_RE = re.compile(r"\\(?:begin|end)\{[a-zA-Z*]+\}")

LATEX_DROP_THRESHOLD = 0.08


def _latex_pollution_ratio(text: str) -> float:
    if not text:
        return 0.0
    matches = (
        len(_LATEX_COMMAND_RE.findall(text))
        + len(_LATEX_SCRIPT_RE.findall(text))
        + len(_LATEX_INLINE_MATH_RE.findall(text))
        + len(_LATEX_ENV_RE.findall(text))
    )
    approx_chars = matches * 6
    return approx_chars / max(1, len(text))


def _strip_latex_noise(text: str) -> str:
    text = _LATEX_ENV_RE.sub(" ", text)
    text = _LATEX_INLINE_MATH_RE.sub(" ", text)
    text = _LATEX_SCRIPT_RE.sub(" ", text)
    text = _LATEX_COMMAND_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def filter_latex_pollution(paragraphs: list[str]) -> list[str]:
    cleaned = []
    dropped = 0
    for p in paragraphs:
        ratio = _latex_pollution_ratio(p)
        if ratio > LATEX_DROP_THRESHOLD:
            dropped += 1
            continue
        if ratio > 0:
            p = _strip_latex_noise(p)
            if len(p) < 20:
                dropped += 1
                continue
        cleaned.append(p)
    if dropped:
        print(f"🧹 Filtre anti-LaTeX: {dropped:,} paragraphe(s) pollué(s) retiré(s)/nettoyé(s) "
              f"sur {len(paragraphs):,} ({dropped / max(1, len(paragraphs)):.1%}).")
    return cleaned


def _load_wikitext_fr_via_datasets(max_chars: int) -> list[str]:
    datasets = _import_datasets_module()
    print(f"📥 Téléchargement de asi/wikitext_fr (config '{WIKI_CONFIG}') via load_dataset...")
    ds = datasets.load_dataset("asi/wikitext_fr", WIKI_CONFIG, split="train")
    paragraphs, total_chars = [], 0
    for row in ds:
        p = row["paragraph"].strip()
        if not p:
            continue
        paragraphs.append(p)
        total_chars += len(p)
        if total_chars >= max_chars:
            break
    return paragraphs


def _load_wikitext_fr_via_zip(max_chars: int) -> list[str]:
    from huggingface_hub import hf_hub_download
    import zipfile

    folder = "wikitext_72" if WIKI_CONFIG == "wikitext-72" else "wikitext_35"
    print(f"📥 Téléchargement direct de {folder}/wiki.zip depuis asi/wikitext_fr...")
    zip_path = hf_hub_download(repo_id="asi/wikitext_fr", repo_type="dataset", filename=f"{folder}/wiki.zip")
    extract_dir = zip_path + "_extracted"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    train_file = None
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if "train" in fname.lower():
                train_file = os.path.join(root, fname)
                break
    if train_file is None:
        raise FileNotFoundError("Fichier d'entraînement introuvable dans l'archive extraite.")

    with open(train_file, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read(max_chars)
    return [p.strip() for p in raw.split("\n") if len(p.strip()) > 20]


def _load_wikimedia_wikipedia_fr(max_chars: int) -> list[str]:
    try:
        from datasets import load_dataset
    except Exception:  # pragma: no cover - graceful fallback when `datasets` is not installed
        load_dataset = None
        import warnings

        warnings.warn(
            "Optional dependency 'datasets' is not available. Functions that rely on it will raise an error if used.\n"
            "Install it with: pip install datasets"
        )
    print("📥 Téléchargement (streaming) de wikimedia/wikipedia (fr) en repli...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    paragraphs, total_chars = [], 0
    for row in ds:
        text = row["text"].strip()
        for p in text.split("\n\n"):
            p = p.strip()
            if len(p) < 50:  # ignore titres/fragments trop courts
                continue
            paragraphs.append(p)
            total_chars += len(p)
        if total_chars >= max_chars:
            break
    return paragraphs


def load_wikipedia_paragraphs(max_chars: int = WIKI_MAX_CHARS) -> list[str]:
    # ⚠️ asi/wikitext_fr désactivé: son chargement via `load_dataset` échoue
    # systématiquement (script de chargement obsolète), ET son repli par
    # téléchargement/extraction de zip a provoqué un SIGKILL/SIGTERM externe
    # (probable OOM) en tournant en parallèle de 2 process GPU. On va
    # directement sur wikimedia/wikipedia, fiable dans toutes nos runs
    # précédentes et streamé (pas d'extraction de zip en mémoire).
    attempts = [
        (lambda: _load_wikimedia_wikipedia_fr(max_chars), "wikimedia/wikipedia (fr)"),
    ]
    for loader, label in attempts:
        try:
            paragraphs = loader()
            print(f"✅ {len(paragraphs):,} paragraphes chargés depuis {label}.")
            paragraphs = filter_latex_pollution(paragraphs)
            return paragraphs
        except Exception as e:
            print(f"⚠️ Échec avec {label} ({e}).")
    print("↪️ Tous les téléchargements ont échoué — utilisation du texte de secours en français embarqué.")
    return [FALLBACK_TEXT_FR]


def build_or_load_french_tokenizer(
    paragraphs: list[str],
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
    cache_path: str = TOKENIZER_CACHE_PATH,
) -> FrenchTokenizerWrapper:
    """Entraîne un tokenizer BPE byte-level (façon GPT-2, donc pas d'OOV possible)
    sur des paragraphes Wikipedia en français — vocabulaire authentiquement
    français, beaucoup plus compact que le vocab anglais de gpt2 (50257 tokens).
    Si un tokenizer entraîné est déjà en cache sur disque, il est rechargé direct.
    Avant ça, tente de le restaurer depuis la sauvegarde externe HF Hub si absent
    localement (ex: nouvelle session Kaggle sans 'Save Version' précédent)."""
    from tokenizers import Tokenizer

    try_restore_backup_from_hf(cache_path, "fr_bpe_tokenizer.json")

    if os.path.exists(cache_path):
        print(f"📂 Tokenizer français rechargé depuis {cache_path}.")
        tok = Tokenizer.from_file(cache_path)
        return FrenchTokenizerWrapper(tok)

    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    print(f"🛠️ Entraînement d'un tokenizer BPE français (vocab_size={vocab_size}) "
          f"sur {len(paragraphs):,} paragraphes...")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator(paragraphs, trainer=trainer)
    # Enregistre <|endoftext|> comme token spécial "atomique": sans ça, encode()
    # découperait la chaîne littérale en octets au lieu de la reconnaître d'un bloc.
    tokenizer.add_special_tokens(["<|endoftext|>"])
    tokenizer.save(cache_path)
    print(f"✅ Tokenizer entraîné (vocabulaire réel: {tokenizer.get_vocab_size()} tokens) "
          f"et sauvegardé dans {cache_path}.")
    # Sauvegarde externe immédiate: le tokenizer ne change plus jamais ensuite,
    # donc un seul push suffit (pas besoin de le répéter à chaque restauration).
    push_backup_to_hf(cache_path, "fr_bpe_tokenizer.json")
    return FrenchTokenizerWrapper(tokenizer)


# ============================================================================
# Chargement du dataset — TinyStories-French, avec repli sur un texte français
# embarqué si le téléchargement échoue (pas de réseau, dataset gated, etc.)
# ============================================================================

FALLBACK_TEXT_FR = """
Il était une fois un petit renard curieux qui vivait à l'orée d'une forêt tranquille.
Chaque matin, il sortait de son terrier pour explorer les sentiers couverts de mousse.
Un jour, il rencontra une chouette sage perchée sur une branche basse.
La chouette lui dit: si tu veux comprendre la forêt, il faut d'abord apprendre à écouter le silence.
Le petit renard s'assit et ferma les yeux. Il entendit le vent dans les feuilles, le ruisseau au loin, et les oiseaux qui chantaient.
Depuis ce jour, il revenait souvent voir la chouette pour apprendre de nouvelles histoires.
Un lapin nommé Noisette vivait aussi dans cette forêt. Il aimait collectionner de petits cailloux ronds.
Chaque cailloux avait une couleur différente, et Noisette les rangeait soigneusement dans un panier tressé.
Un matin de printemps, la rivière déborda légèrement à cause de la fonte des neiges.
Le renard et le lapin décidèrent de construire un petit pont avec des branches pour aider leurs amis à traverser.
Ensemble, ils travaillèrent toute la journée, portant des bâtons et les attachant avec des lianes solides.
Quand le pont fut terminé, tous les animaux de la forêt vinrent le remercier avec des fleurs et des fruits.
La chouette sage regarda la scène depuis son arbre et sourit: la coopération est la plus belle des forces.
Le soir venu, les étoiles apparurent une à une dans le ciel violet, et la forêt s'endormit doucement.
Le lendemain, le petit renard raconta cette aventure à tous ses amis, encore et encore, avec des étoiles dans les yeux.
""" * 40  # répété pour donner un corpus d'entraînement de taille suffisante


def _fix_mojibake(text: str) -> str:
    if "Ã©" in text or "Ã¨" in text or "â€™" in text:
        try:
            return text.encode("latin1").decode("utf8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def load_training_text(n_stories: int = N_STORIES) -> str:
    try:
        from datasets import load_dataset
        print(f"📥 Téléchargement de TinyStories-French...")
        ds = load_dataset("iproskurina/TinyStories-French", split="train")
        column = "french-tinystories" if "french-tinystories" in ds.column_names else ds.column_names[0]
        texts = [t for t in ds[column] if t and t.strip()]
        texts = texts[:n_stories] if n_stories < len(texts) else texts
        texts = [_fix_mojibake(t) for t in texts]
        dataset_text = "\n<|endoftext|>\n".join(texts)
        print(f"✅ Dataset chargé: {len(texts)} histoires en français, {len(dataset_text):,} caractères.")
        if len(texts) < 1500:
            print("ℹ️ Corpus restreint (~1000 histoires) — le modèle reverra plusieurs fois "
                  "les mêmes textes sur 1000 steps, ce qui reste adapté à un modèle de cette taille.")
        return dataset_text
    except Exception as e:
        print(f"⚠️ Impossible de charger TinyStories-French ({e}).")
        print("↪️ Utilisation du texte de secours en français (corpus embarqué).")
        return FALLBACK_TEXT_FR


# ============================================================================
# Sources de pré-entraînement à grande échelle (plusieurs milliards de tokens)
# ============================================================================

def _stream_wikipedia_fr_docs():
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    for row in ds:
        text = row["text"].strip()
        if text:
            yield text


def _stream_fineweb2_fr_docs():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-2", "fra_Latn", split="train", streaming=True)
    for row in ds:
        text = row["text"].strip()
        if text:
            yield text


def _stream_oscar_fr_docs():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN absent — OSCAR-2301 nécessite d'accepter les conditions d'accès "
            "sur huggingface.co/datasets/oscar-corpus/OSCAR-2301 puis d'exporter un token."
        )
    from datasets import load_dataset
    ds = load_dataset("oscar-corpus/OSCAR-2301", language="fr", split="train", streaming=True, token=token)
    for row in ds:
        text = row["text"].strip()
        if text:
            yield text


PRETRAIN_SOURCES = [
    {"name": "Wikipedia FR (wikimedia/wikipedia)",              "weight": 0.15, "make_gen": _stream_wikipedia_fr_docs},
    {"name": "FineWeb-2 FR (HuggingFaceFW/fineweb-2, fra_Latn)", "weight": 0.70, "make_gen": _stream_fineweb2_fr_docs},
    {"name": "OSCAR-2301 FR (opportuniste, requiert HF_TOKEN)",  "weight": 0.15, "make_gen": _stream_oscar_fr_docs},
]


def build_pretraining_corpus_bin(
    tokenizer: FrenchTokenizerWrapper,
    target_tokens: int = TARGET_PRETRAIN_TOKENS,
    bin_path: str = PRETRAIN_BIN_PATH,
    meta_path: str = PRETRAIN_META_PATH,
    extra_text_once: str = "",
) -> tuple[str, int]:
    dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32
    print("cwd:", os.getcwd())
    print("bin existe:", os.path.exists(PRETRAIN_BIN_PATH), PRETRAIN_BIN_PATH)
    print("meta existe:", os.path.exists(PRETRAIN_META_PATH), PRETRAIN_META_PATH)
    if os.path.exists(bin_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("target_tokens") == target_tokens and meta.get("vocab_size") == tokenizer.vocab_size:
            total = meta["total_tokens"]
            print(f"📂 Corpus de pré-entraînement déjà construit sur Drive ({total:,} tokens) — réutilisation.")
            return bin_path, total
        print("⚠️ Corpus existant sur Drive mais cible ou vocabulaire différents — reconstruction complète.")

    print(f"🏗️ Construction du corpus de pré-entraînement (~{target_tokens / 1e9:.2f}G tokens visés) "
          f"depuis {len(PRETRAIN_SOURCES)} sources...")

    stats = {src["name"]: 0 for src in PRETRAIN_SOURCES}
    active = []
    for src in PRETRAIN_SOURCES:
        try:
            gen = src["make_gen"]()
            active.append({"name": src["name"], "weight": src["weight"], "gen": gen})
            print(f"   ✅ Source active: {src['name']} (poids {src['weight']:.0%})")
        except Exception as e:
            print(f"   ⚠️ Source indisponible dès l'initialisation, ignorée: {src['name']} ({e})")

    if not active and not extra_text_once:
        raise RuntimeError("Aucune source de pré-entraînement disponible — vérifie ta connexion et 'datasets'.")

    eot = tokenizer.eot_token
    total_tokens = 0
    buffer: list[int] = []
    FLUSH_EVERY = 500_000
    PROGRESS_EVERY = 50_000_000
    HEARTBEAT_SECONDS = 30
    _heartbeat_start = time.time()
    _last_heartbeat = _heartbeat_start
    _tokens_at_last_heartbeat = 0

    with open(bin_path, "wb") as f:
        def flush():
            nonlocal buffer
            if buffer:
                np.array(buffer, dtype=dtype).tofile(f)
                buffer = []

        if extra_text_once:
            ids = tokenizer.encode(extra_text_once, allowed_special="all")
            buffer.extend(ids)
            total_tokens += len(ids)

        last_progress = 0
        while total_tokens < target_tokens and active:
            weights = [s["weight"] for s in active]
            src = random.choices(active, weights=weights, k=1)[0]
            try:
                doc = next(src["gen"])
            except StopIteration:
                print(f"   ↪️ Source épuisée: {src['name']} ({stats[src['name']]:,} tokens fournis).")
                active.remove(src)
                continue
            except Exception as e:
                print(f"   ⚠️ Erreur sur {src['name']}, source retirée: {e}")
                active.remove(src)
                continue

            ids = tokenizer.encode(doc, allowed_special="all")
            if not ids:
                continue
            buffer.extend(ids)
            buffer.append(eot)
            stats[src["name"]] += len(ids) + 1
            total_tokens += len(ids) + 1

            if len(buffer) >= FLUSH_EVERY:
                flush()
            if total_tokens - last_progress >= PROGRESS_EVERY:
                last_progress = total_tokens
                print(f"   … {total_tokens / 1e9:.3f}G / {target_tokens / 1e9:.2f}G tokens "
                      f"({total_tokens / target_tokens:.1%})")

            now = time.time()
            if now - _last_heartbeat >= HEARTBEAT_SECONDS:
                rate = (total_tokens - _tokens_at_last_heartbeat) / max(1e-6, now - _last_heartbeat)
                elapsed_min = (now - _heartbeat_start) / 60
                print(f"   💓 toujours actif (t+{elapsed_min:.1f} min) — "
                      f"{total_tokens:,} tokens au total, ~{rate:,.0f} tokens/s sur les "
                      f"{HEARTBEAT_SECONDS}s écoulées, source actuelle: {src['name']}")
                _last_heartbeat = now
                _tokens_at_last_heartbeat = total_tokens

        flush()

    if total_tokens < target_tokens:
        print(f"⚠️ Cible non atteinte: {total_tokens:,} / {target_tokens:,} tokens "
              f"(toutes les sources disponibles ont été épuisées).")

    print("📊 Répartition finale par source:")
    for name, n in stats.items():
        print(f"   - {name}: {n:,} tokens ({n / max(1, total_tokens):.1%})")

    meta = {
        "target_tokens": target_tokens,
        "total_tokens": total_tokens,
        "vocab_size": tokenizer.vocab_size,
        "dtype": str(np.dtype(dtype)),
        "sources": stats,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return bin_path, total_tokens


# ============================================================================
# Trainer
# ============================================================================

class LLMTrainer:
    def __init__(
        self,
        model: ModernLLM,
        args: ModelArgs,
        tokenizer: FrenchTokenizerWrapper,
        accelerator: Accelerator,
        dataset_text: str | None = None,
        bin_path: str | None = None,
        total_tokens: int | None = None,
        val_fraction: float = 0.05,
        val_tokens_cap: int = VAL_TOKENS_CAP,
        raw_model: ModernLLM | None = None,
    ):
        self.accelerator = accelerator
        self.model = model
        self.raw_model = raw_model or model
        self.args = args
        self.tokenizer = tokenizer

        if bin_path is not None:
            assert total_tokens is not None, "total_tokens requis quand bin_path est fourni"
            dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32
            val_tokens = min(int(total_tokens * val_fraction), val_tokens_cap)
            split_idx = total_tokens - val_tokens
            self.train_dataset = TextDatasetMemmap(bin_path, dtype, 0, split_idx, args.max_seq_len)
            self.val_dataset = (
                TextDatasetMemmap(bin_path, dtype, split_idx, total_tokens, args.max_seq_len)
                if val_tokens > args.max_seq_len + 1 else self.train_dataset
            )
            if accelerator.is_main_process:
                print(f"🔤 Corpus (memmap, sur disque): {total_tokens:,} tokens — "
                      f"train {split_idx:,} / val {val_tokens:,}.")
        else:
            assert dataset_text is not None, "dataset_text ou bin_path doit être fourni"
            token_ids = self.tokenizer.encode(dataset_text, allowed_special="all")
            if accelerator.is_main_process:
                print(f"🔤 Corpus tokenisé: {len(token_ids):,} tokens.")

            split_idx = int(len(token_ids) * (1 - val_fraction))
            train_ids = token_ids[:split_idx]
            val_ids = token_ids[split_idx:]

            self.train_dataset = TextDataset(train_ids, block_size=args.max_seq_len)
            self.val_dataset = TextDataset(val_ids, block_size=args.max_seq_len) if len(val_ids) > args.max_seq_len + 1 else self.train_dataset

        decay_params, no_decay_params = [], []
        seen = set()
        for name, p in self.raw_model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay_params if p.dim() >= 2 else no_decay_params).append(p)

        optim_groups = [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and args.device.startswith("cuda")
        self.optimizer = torch.optim.AdamW(
            optim_groups, lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8, fused=use_fused,
        )
        self.grad_clip = 1.0

    def get_lr(self, step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        if step >= max_steps:
            return min_lr
        decay_ratio = min(max((step - warmup_steps) / max(1, (max_steps - warmup_steps)), 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        # accelerator.accumulate() gère automatiquement l'accumulation sur
        # GRAD_ACCUM_STEPS micro-steps: zero_grad()/optimizer.step() ci-dessous
        # sont bien appelés à chaque micro-step dans le code, mais Accelerate
        # les rend effectifs seulement au dernier micro-step du groupe (et
        # désactive la synchronisation DDP entre GPU sur les micro-steps
        # intermédiaires, pour ne pas payer ce coût réseau à chaque micro-step).
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad(set_to_none=True)
            with self.accelerator.autocast():
                logits, loss = self.model(x, targets=y)

            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, n_batches: int = 20, batch_size: int = BATCH_SIZE) -> float:
        self.raw_model.eval()
        losses = []
        for _ in range(n_batches):
            idxs = [random.randint(0, len(self.val_dataset) - 1) for _ in range(min(batch_size, max(1, len(self.val_dataset))))]
            xb = torch.stack([self.val_dataset[i][0] for i in idxs])
            yb = torch.stack([self.val_dataset[i][1] for i in idxs])
            xb, yb = xb.to(self.args.device), yb.to(self.args.device)
            _, loss = self.raw_model(xb, targets=yb)
            losses.append(loss.item())
        self.raw_model.train()
        return sum(losses) / len(losses)


# ============================================================================
# Point d'entrée — entraînement complet (multi-GPU via accelerate + torch.multiprocessing.spawn)
# ============================================================================

def prepare_data_once() -> tuple[str, int]:
    """Prépare tokenizer + corpus AVANT le lancement des processus GPU (mono ou
    multi). Avant, cette étape se faisait à l'intérieur de main_worker (rang 0
    pendant que le rang 1 attendait, GPU + Accelerate déjà initialisés sur les
    deux) — inutile puisque cette phase est 100% réseau/CPU, et ça ajoutait de
    la pression mémoire hôte pendant la fenêtre la plus fragile (observé:
    SIGTERM externe, probable OOM, pendant le téléchargement Wikipedia avec 2
    process GPU déjà vivants). Maintenant: un seul process Python tourne
    pendant toute cette phase, les process GPU ne démarrent qu'une fois le
    corpus prêt sur disque."""
    wiki_paragraphs = load_wikipedia_paragraphs()
    tokenizer = build_or_load_french_tokenizer(wiki_paragraphs, vocab_size=TOKENIZER_VOCAB_SIZE)
    tinystories_text = load_training_text(N_STORIES)
    wiki_sample_text = "\n<|endoftext|>\n".join(wiki_paragraphs)
    seed_text = tinystories_text + "\n<|endoftext|>\n" + wiki_sample_text
    bin_path, total_tokens = build_pretraining_corpus_bin(
        tokenizer, target_tokens=TARGET_PRETRAIN_TOKENS, extra_text_once=seed_text,
    )
    return bin_path, total_tokens


def main_worker(rank: int, world_size: int, bin_path: str, total_tokens: int) -> None:

    if world_size > 1:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        # ⚠️ Ce désactivage NCCL P2P/IB reste conditionné à `_is_kaggle()` —
        # il ne s'applique donc PAS sur un poste avec des RTX A5000 (hors
        # Kaggle). C'était un correctif pour la topologie spécifique du T4x2
        # Kaggle (pas de lien P2P réel entre les 2 GPU). Sur un poste A5000
        # multi-GPU, vérifie ta topologie réelle avec `nvidia-smi topo -m`
        # avant de désactiver P2P/IB toi-même — le désactiver alors qu'un vrai
        # lien P2P/NVLink existe dégraderait les perfs au lieu de les corriger.
        if _is_kaggle():
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")

        # ⚠️ Avec torch.multiprocessing.spawn (plutôt que `accelerate launch` ou
        # `notebook_launcher`), la détection automatique du contexte distribué
        # par Accelerator() peut échouer silencieusement — chaque processus
        # spawné retombe alors en mode "solo" (is_main_process=True sur TOUS
        # les rangs, aucune vraie synchronisation de gradients entre GPU).
        # Constaté en pratique : logs entièrement dupliqués sur des print()
        # protégés par `is_main_process`. On initialise donc le process group
        # PyTorch EXPLICITEMENT ici, avant Accelerator(), plutôt que de
        # compter sur sa détection automatique.
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend="nccl", rank=rank, world_size=world_size,
        )

    torch.manual_seed(SEED)
    random.seed(SEED + rank)

    # Le corpus est déjà prêt sur disque (voir prepare_data_once, appelée avant
    # mp.spawn) — chaque rang recharge juste le tokenizer depuis le cache local
    # (quasi instantané, aucun appel réseau nécessaire à ce stade).
    wiki_paragraphs_for_tokenizer_cache = []  # non utilisé: le tokenizer est déjà en cache
    tokenizer = build_or_load_french_tokenizer(wiki_paragraphs_for_tokenizer_cache, vocab_size=TOKENIZER_VOCAB_SIZE)

    mixed_precision = "no"
    if torch.cuda.is_available():
        mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"  # T4 (Kaggle) -> fp16, pas de bf16
    accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    device = str(accelerator.device)

    if world_size > 1 and accelerator.is_main_process:
        print(f"🔍 Vérification synchro DDP : num_processes={accelerator.num_processes}, "
              f"is_main_process(rang {rank})={accelerator.is_main_process} — "
              f"si tu vois ce message imprimé plus d'une fois, la synchronisation "
              f"est encore cassée.")

    if accelerator.is_main_process:
        print("\n🚀 Initialisation du modèle...")
    args = ModelArgs(vocab_size=tokenizer.vocab_size, device=device)
    model = ModernLLM(args)
    n_params = model.num_params()
    if accelerator.is_main_process:
        print(f"✅ Modèle instancié sur {accelerator.num_processes} GPU(s) avec {n_params / 1e6:.2f}M paramètres "
              f"(vocab={args.vocab_size}, dim={args.dim}, n_layers={args.n_layers}, precision={mixed_precision}).")

    # ⚠️ torch.compile: gain de throughput généralement net sur Ampere+ (le
    # backend Inductor/Triton cible bien cette génération de GPU), alors que
    # sur T4 (Turing) le gain est souvent marginal voire négatif à cause d'un
    # support moins mature. Compilé APRÈS le chargement d'un éventuel
    # checkpoint (juste au-dessus) mais AVANT accelerator.prepare(), pour que
    # le state_dict se charge sur le module non compilé (évite les soucis de
    # préfixes de clés "_orig_mod." que torch.compile peut introduire). Le 1er
    # step sera plus lent (compilation à froid) — c'est normal et attendu.
    if torch.cuda.is_available():
        model = torch.compile(model)

    # 🛟 Avant de repartir de poids aléatoires: tente d'abord une restauration
    # depuis la sauvegarde externe HF Hub (utile après une session Kaggle
    # perdue sans 'Save Version'), PUIS le comportement RESUME_FROM_CHECKPOINT
    # habituel sur le fichier local (potentiellement fraîchement restauré).
    if accelerator.is_main_process and RESUME_FROM_CHECKPOINT and not os.path.exists(CHECKPOINT_PATH):
        try_restore_backup_from_hf(CHECKPOINT_PATH, "best_model.pt")
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()  # tous les rangs attendent la restauration éventuelle du rang 0

    resumed_step = 0
    resumed_step_is_approximate = False
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        _ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(_ckpt["model_state_dict"])
            _tokens_seen = _ckpt.get("tokens_seen")
            if _tokens_seen:
                # Robuste à tout changement futur de BATCH_SIZE/GRAD_ACCUM_STEPS:
                # le nombre de tokens vus ne dépend pas du découpage en steps.
                # tokens_per_micro_step n'est pas encore calculé à ce point du
                # code (dépend de accelerator.num_processes, déjà connu) donc on
                # le recalcule ici directement.
                _tokens_per_micro_step = BATCH_SIZE * args.max_seq_len * accelerator.num_processes
                resumed_step = int(_tokens_seen // (_tokens_per_micro_step * GRAD_ACCUM_STEPS))
            else:
                # Checkpoint antérieur à l'ajout de "tokens_seen" (avant
                # l'introduction de l'accumulation de gradient): à l'époque,
                # "step" comptait les micro-steps 1:1. Le réutiliser tel quel
                # comme step réel surestimerait la progression de GRAD_ACCUM_STEPS
                # fois — on le convertit donc, avec un avertissement explicite
                # car cette conversion est une approximation ponctuelle,
                # seulement pour ce cas de transition.
                _old_step = int(_ckpt.get("step", 0) or 0)
                resumed_step = _old_step // GRAD_ACCUM_STEPS
                resumed_step_is_approximate = _old_step > 0
            if accelerator.is_main_process:
                print(f"♻️  Reprise depuis '{CHECKPOINT_PATH}' (au lieu de repartir de poids aléatoires)...")
                print(f"   ↳ poids chargés (step réel estimé: {resumed_step}, "
                      f"val_loss précédent: {_ckpt.get('val_loss', float('nan')):.4f})")
                if resumed_step_is_approximate:
                    print(f"   ↳ ⚠️ Checkpoint antérieur à l'ajout de l'accumulation de gradient — "
                          f"step converti par approximation (÷{GRAD_ACCUM_STEPS}), pas une valeur exacte.")
        except RuntimeError as e:
            # ⚠️ Le checkpoint existe mais correspond à une architecture
            # différente de ModelArgs actuel (ex: on a changé dim/n_layers pour
            # passer à une taille de modèle différente sans renommer l'ancien
            # best_model.pt). Plutôt que de planter tout le run multi-GPU pour
            # ça (déjà arrivé plusieurs fois), on ignore le checkpoint et on
            # repart de poids aléatoires — avec un avertissement explicite,
            # pour que ça ne passe jamais inaperçu.
            if accelerator.is_main_process:
                print(f"⚠️  '{CHECKPOINT_PATH}' existe mais ne correspond pas à l'architecture actuelle "
                      f"(dim={args.dim}, n_layers={args.n_layers}) — probablement un checkpoint d'une "
                      f"AUTRE taille de modèle. Poids ALÉATOIRES utilisés pour ce run (le fichier "
                      f"n'est pas touché). Renomme/déplace-le si ce n'est pas voulu.\n   Détail: {e}")

    trainer = LLMTrainer(
        model, args, tokenizer=tokenizer, accelerator=accelerator,
        bin_path=bin_path, total_tokens=total_tokens,
    )
    if accelerator.is_main_process:
        print(f"📚 Fenêtres d'entraînement: {len(trainer.train_dataset):,} | validation: {len(trainer.val_dataset):,}")

    if isinstance(trainer.train_dataset, TextDatasetMemmap):
        n = accelerator.num_processes
        r = accelerator.process_index
        full_start, full_end = trainer.train_dataset.start, trainer.train_dataset.end
        shard_size = (full_end - full_start) // n
        shard_start = full_start + r * shard_size
        shard_end = full_end if r == n - 1 else shard_start + shard_size
        iterable_train_dataset = RandomWindowIterableDataset(
            trainer.train_dataset.bin_path, trainer.train_dataset.dtype,
            shard_start, shard_end, args.max_seq_len, seed=SEED + r,
        )
        # ⚠️ num_workers=2 + pin_memory=True: avec BATCH_SIZE remonté et un GPU
        # nettement plus rapide (A5000 vs T4), num_workers=0 (valeur par
        # défaut, utilisée dans la version T4 de ce script) devient un goulot
        # d'étranglement CPU — le GPU finit par attendre les données.
        # RandomWindowIterableDataset ouvre déjà son memmap paresseusement par
        # worker (_ensure_mmap), donc num_workers>0 fonctionne nativement sans
        # modification. pin_memory accélère le transfert host->device;
        # persistent_workers évite de recréer les workers à chaque itération
        # de infinite_loader (utile car ce dataset est un IterableDataset
        # infini qui ne "termine" jamais vraiment son epoch).
        loader = torch.utils.data.DataLoader(
            iterable_train_dataset, batch_size=BATCH_SIZE, drop_last=True,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )
    else:
        loader = torch.utils.data.DataLoader(
            trainer.train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )

    model, trainer.optimizer, loader = accelerator.prepare(model, trainer.optimizer, loader)
    trainer.model = model
    trainer.raw_model = accelerator.unwrap_model(model)
    raw_model = trainer.raw_model

    tokens_per_micro_step = BATCH_SIZE * args.max_seq_len * accelerator.num_processes
    micro_steps_total = max(1, total_tokens // tokens_per_micro_step)
    # effective_max_steps = nombre de vraies MISES À JOUR DE POIDS (ce qui
    # compte pour le planning LR), pas le nombre brut de micro-steps — avec
    # GRAD_ACCUM_STEPS=4, il faut 4 micro-steps pour une seule mise à jour.
    effective_max_steps = max(1, micro_steps_total // GRAD_ACCUM_STEPS)
    effective_warmup_steps = max(100, int(0.05 * effective_max_steps))
    if accelerator.is_main_process:
        effective_batch = BATCH_SIZE * accelerator.num_processes * GRAD_ACCUM_STEPS
        print(f"🧮 {total_tokens:,} tokens / {tokens_per_micro_step:,} tokens par micro-step "
              f"(batch micro {BATCH_SIZE * accelerator.num_processes} = {BATCH_SIZE} x {accelerator.num_processes} GPU, "
              f"x{GRAD_ACCUM_STEPS} accumulation = batch effectif {effective_batch}) "
              f"= {effective_max_steps:,} steps réels (mises à jour de poids) pour ~1 epoch "
              f"({micro_steps_total:,} micro-steps au total).")

    def infinite_loader(dl):
        while True:
            for batch in dl:
                yield batch

    data_iter = infinite_loader(loader)

    best_val_loss = float(_ckpt.get("val_loss", float("inf"))) if resumed_step else float("inf")
    _last_latest_backup = time.time()
    # ⚠️ Throttle RETIRÉ : il causait un problème pire que celui qu'il évitait.
    # En repo PRIVÉ, on avait dépassé le quota de stockage (403) à force de
    # pousser un fichier de ~787 Mo à chaque amélioration. Mais throttler
    # signifiait aussi que si la session Kaggle mourait AVANT la prochaine
    # fenêtre de push, le meilleur modèle local (le plus récent) ne partait
    # jamais vers HF Hub — constaté en pratique : restauration d'un checkpoint
    # à des milliers de steps de retard sur la vraie progression. Le repo
    # étant maintenant PUBLIC (quota bien plus généreux), on repousse "best"
    # immédiatement à chaque amélioration, sans throttle.
    start_step = min(resumed_step, effective_max_steps - 1) if resumed_step else 0
    start_micro_step = start_step * GRAD_ACCUM_STEPS
    if accelerator.is_main_process and start_step > 0:
        print(f"⏩ Reprise à step {start_step:,}/{effective_max_steps:,} (au lieu de 0) — "
              f"le planning LR reprend là où il en était, pas de re-warmup depuis zéro.")

    if accelerator.is_main_process:
        print(f"\n🏋️ Entraînement pour {effective_max_steps:,} steps réels "
              f"({micro_steps_total:,} micro-steps, accumulation x{GRAD_ACCUM_STEPS})...\n")

    for micro_step in range(start_micro_step, micro_steps_total):
        step = micro_step // GRAD_ACCUM_STEPS  # step réel (mise à jour de poids)
        is_update_boundary = (micro_step + 1) % GRAD_ACCUM_STEPS == 0

        lr = trainer.get_lr(step, effective_max_steps, effective_warmup_steps, MAX_LR, MIN_LR)
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr

        x_batch, y_batch = next(data_iter)
        train_loss = trainer.train_step(x_batch, y_batch)  # micro-step; poids mis à jour seulement si is_update_boundary

        if not is_update_boundary:
            continue  # logs/évals/sauvegardes uniquement sur une vraie mise à jour de poids

        if accelerator.is_main_process and (step % 20 == 0 or step == effective_max_steps - 1):
            print(f"step {step:06d}/{effective_max_steps} | lr {lr:.2e} | train_loss {train_loss:.4f}")

        if accelerator.is_main_process and (step % EVAL_INTERVAL == 0 or step == effective_max_steps - 1):
            val_loss = trainer.evaluate()
            print(f"   ↳ 📊 val_loss {val_loss:.4f} (best: {best_val_loss:.4f})")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model_state_dict": raw_model.state_dict(), "args": args, "step": step, "val_loss": val_loss, "tokens_seen": (step + 1) * tokens_per_micro_step * GRAD_ACCUM_STEPS},
                    CHECKPOINT_PATH,
                )
                print(f"   ↳ 💾 Nouveau meilleur modèle sauvegardé ({CHECKPOINT_PATH})")
                # 🛟 Poussé immédiatement hors de Kaggle (repo public, pas de
                # throttle) — survit à un crash/coupure même si la session
                # meurt la seconde suivante.
                push_backup_to_hf(CHECKPOINT_PATH, "best_model.pt")

        # 🛟 Filet de sécurité indépendant des améliorations de val_loss: repousse
        # l'état ACTUEL toutes les HF_BACKUP_LATEST_INTERVAL_SECONDS, même sans
        # nouveau record, pour ne jamais perdre plus de ~10 min de progression.
        if accelerator.is_main_process and HF_BACKUP_ENABLED:
            now = time.time()
            if now - _last_latest_backup >= HF_BACKUP_LATEST_INTERVAL_SECONDS:
                torch.save(
                    {"model_state_dict": raw_model.state_dict(), "args": args, "step": step, "val_loss": None, "tokens_seen": (step + 1) * tokens_per_micro_step * GRAD_ACCUM_STEPS},
                    LATEST_CHECKPOINT_PATH,
                )
                push_backup_to_hf(LATEST_CHECKPOINT_PATH, "latest_model.pt")
                _last_latest_backup = now

        if accelerator.is_main_process and step % GEN_INTERVAL == 0 and step > 0:
            sample = raw_model.generate(
                prompt="Il était une fois",
                tokenizer=trainer.tokenizer,
                max_new_tokens=40,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.3,
            )
            print(f"   ↳ 📝 Échantillon: {sample!r}\n")

    if accelerator.is_main_process:
        print("\n✅ Entraînement terminé.")

        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            raw_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"📂 Meilleur modèle rechargé (step {checkpoint['step']}, val_loss {checkpoint['val_loss']:.4f})")

        print("\n🧪 Génération finale (temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.3):")
        final_sample = raw_model.generate(
            prompt="Il était une fois",
            tokenizer=trainer.tokenizer,
            max_new_tokens=80,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.3,
        )
        print(final_sample)


if __name__ == "__main__":
    # Préparation des données AVANT tout process GPU — voir prepare_data_once()
    # pour le pourquoi (mémoire hôte, fenêtre de crash observée).
    _bin_path, _total_tokens = prepare_data_once()

    if WORLD_SIZE > 1:
        import sys
        if not hasattr(sys.modules["__main__"], "__spec__"):
            sys.modules["__main__"].__spec__ = None

        print(f"🖥️ {WORLD_SIZE} GPU détectés — lancement en DistributedDataParallel via "
              f"torch.multiprocessing.spawn ({WORLD_SIZE} processus, un par GPU, méthode "
              f"'spawn' — la seule sûre avec CUDA, contrairement à notebook_launcher qui "
              f"utilise 'fork').")
        mp.spawn(main_worker, args=(WORLD_SIZE, _bin_path, _total_tokens), nprocs=WORLD_SIZE, join=True)
    else:
        main_worker(rank=0, world_size=1, bin_path=_bin_path, total_tokens=_total_tokens)
