import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertModel,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_PROFANITY = [
    "fuck", "f**k", "f**ker", "motherfucker", "motherfucking",
    "shit", "shitty", "bitch", "bitches", "bastard",
    "asshole", "ass", "arse", "crap", "dick", "d!ck", "d1ck",
    "cock", "c0ck", "cum", "whore", "slut", "darn", "damn",
    "bloody", "prick", "twat", "screw", "sucker", "moron", "retard",
    "nigger", "nigga", "cunt", "jackass", "dumbass", "bullshit",
]

SEMANTIC_TOXIC_SEEDS = [
    "idiot", "stupid", "dumb", "jerk", "loser", "trash", "garbage",
    "scum", "filth", "disgusting", "pathetic", "worthless", "useless",
    "horrible", "awful", "terrible", "hateful", "vile", "despicable",
]

HOMOGLYPH_MAP = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0445": "x", "\u0456": "i", "\u0455": "s",
    "\u0458": "j", "\u0501": "d", "\u051b": "q",
    "\u03b1": "a", "\u03b2": "b", "\u03b3": "y", "\u03b4": "d",
    "\u03b5": "e", "\u03b6": "z", "\u03b7": "n", "\u03b9": "i",
    "\u03ba": "k", "\u03bc": "m", "\u03bd": "n", "\u03bf": "o",
    "\u03c1": "p", "\u03c4": "t", "\u03c5": "u", "\u03c7": "x",
    "\u03c9": "w",
    "\uff10": "0", "\uff11": "1", "\uff12": "2", "\uff13": "3",
    "\uff14": "4", "\uff15": "5", "\uff16": "6", "\uff17": "7",
    "\uff18": "8", "\uff19": "9",
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
})


def normalize_unicode(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(HOMOGLYPH_MAP)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", "\r")
    )
    return text


LEET_MAP = {
    "a": ["@", "4"], "b": ["8"], "c": ["(", "k"], "d": ["|)"],
    "e": ["3"], "i": ["1", "!", "l"], "k": ["|<"],
    "l": ["1", "|"], "o": ["0"], "s": ["$", "5"],
    "t": ["7", "+"], "u": ["v", "o"], "y": ["\u00a5"],
}


def random_leet_replace(word: str, prob: float = 0.4) -> str:
    return "".join(
        random.choice(LEET_MAP[c.lower()])
        if c.lower() in LEET_MAP and random.random() < prob
        else c
        for c in word
    )


def inject_punctuations(word: str, prob: float = 0.25) -> str:
    puncts = [".", "*", "@", "!", "#", "$", "%", "^", "&", "-"]
    out = []
    for ch in word:
        out.append(ch)
        if ch.strip() and random.random() < prob:
            out.append(random.choice(puncts))
    return "".join(out)


def drop_vowels(word: str, prob: float = 0.3) -> str:
    return "".join(
        ch for ch in word
        if not (ch.lower() in "aeiou" and random.random() < prob)
    )


def repeat_chars(word: str, prob: float = 0.2) -> str:
    out = []
    for ch in word:
        out.append(ch)
        if ch.strip() and random.random() < prob:
            out.append(ch * random.randint(1, 3))
    return "".join(out)


def insert_homoglyphs(word: str, prob: float = 0.3) -> str:
    reverse_map = {
        "a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440",
        "c": "\u0441", "x": "\u0445", "i": "\u0456", "s": "\u0455",
    }
    return "".join(
        reverse_map[c.lower()]
        if c.lower() in reverse_map and random.random() < prob
        else c
        for c in word
    )


def generate_obfuscations(word: str, n: int = 14) -> List[str]:
    variants = set()
    attempts = 0
    while len(variants) < n and attempts < n * 8:
        s = word
        if random.random() < 0.75:
            s = random_leet_replace(s, prob=random.uniform(0.2, 0.7))
        if random.random() < 0.55:
            s = inject_punctuations(s, prob=random.uniform(0.1, 0.4))
        if random.random() < 0.45:
            s = repeat_chars(s, prob=random.uniform(0.05, 0.35))
        if random.random() < 0.45:
            s = drop_vowels(s, prob=random.uniform(0.15, 0.5))
        if random.random() < 0.35:
            s = insert_homoglyphs(s, prob=random.uniform(0.2, 0.6))
        variants.add(s)
        attempts += 1
    variants.add(word)
    return list(variants)


def dot_separate(word: str, sep: str = ".") -> str:
    return sep.join(word)


def generate_dot_variants(word: str) -> List[str]:
    seps = [".", "-", "*", "_", "~", " "]
    variants = [dot_separate(word, s) for s in seps]
    variants.append(dot_separate(word.upper(), "."))
    variants.append(dot_separate(word.capitalize(), "-"))
    return variants


SYNONYM_MAP = {
    "stupid": ["dumb", "idiotic", "moronic", "brainless", "dense"],
    "idiot": ["moron", "imbecile", "fool", "dimwit", "blockhead"],
    "hate": ["despise", "loathe", "detest", "abhor"],
    "terrible": ["awful", "dreadful", "horrible", "atrocious"],
    "ugly": ["hideous", "repulsive", "grotesque", "revolting"],
    "jerk": ["creep", "bully", "clown", "punk"],
    "trash": ["garbage", "filth", "scum", "waste"],
    "loser": ["failure", "flop", "washout", "nobody"],
    "awful": ["terrible", "dreadful", "appalling", "atrocious"],
    "pathetic": ["pitiful", "hopeless", "feeble", "lamentable"],
}


def synonym_augment_sentence(sentence: str, n: int = 3) -> List[str]:
    words = sentence.split()
    results = []
    for _ in range(n):
        new_words = []
        for w in words:
            lower = w.lower().strip(".,!?;:")
            if lower in SYNONYM_MAP and random.random() < 0.6:
                new_words.append(random.choice(SYNONYM_MAP[lower]))
            else:
                new_words.append(w)
        results.append(" ".join(new_words))
    return results


KEYBOARD_NEIGHBORS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko",
    "j": "huikmn", "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm",
    "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awedxz",
    "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc",
    "y": "tghu", "z": "asx",
}

REALWORLD_TOXIC_COLUMNS = (
    "toxic", "severe_toxic", "obscene", "threat", "insult",
    "identity_hate", "toxicity", "severe_toxicity",
    "identity_attack", "sexual_explicit",
)

IDENTITY_TERMS = [
    "black", "white", "asian", "jewish", "muslim", "christian",
    "gay", "trans", "woman", "women", "immigrant", "disabled",
    "minority",
]


def _safe_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z'_-]*", normalize_unicode(text))


def _bug_word(word: str) -> str:
    if len(word) <= 2:
        return word

    op = random.choice(
        ("insert", "delete", "swap", "substitute", "repeat", "leet", "space")
    )
    chars = list(word)
    idx = random.randrange(len(chars))

    if op == "insert":
        chars.insert(idx, random.choice("abcdefghijklmnopqrstuvwxyz"))
    elif op == "delete" and len(chars) > 3:
        del chars[idx]
    elif op == "swap" and idx < len(chars) - 1:
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif op == "substitute":
        c = chars[idx].lower()
        chars[idx] = random.choice(KEYBOARD_NEIGHBORS.get(c, "abcdefghijklmnopqrstuvwxyz"))
    elif op == "repeat":
        chars[idx] = chars[idx] * random.randint(2, 4)
    elif op == "leet":
        return random_leet_replace(word, prob=0.65)
    elif op == "space":
        return " ".join(chars)

    return "".join(chars)


def textbugger_augment(text: str, severity: float = 0.18) -> str:
    text = normalize_unicode(text)
    if not text:
        return text

    words = _safe_words(text)
    if not words:
        return text

    n_changes = max(1, int(round(len(words) * severity)))
    candidates = [w for w in words if len(w) > 2 and not w.startswith("http")]
    if not candidates:
        return text

    selected = set(random.sample(candidates, k=min(n_changes, len(candidates))))

    def repl(match: re.Match) -> str:
        word = match.group(0)
        return _bug_word(word) if word in selected else word

    return re.sub(r"[A-Za-z][A-Za-z'_-]*", repl, text)


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(https?://\S+|www\.[^\s]+)")
HANDLE_RE = re.compile(r"(?<!\w)@\w[-\w.]*")

SAFE_SYMBOL_PATTERNS = [
    re.compile(r"\bC\+\+\b"),
    re.compile(r"\bNode\.js\b", re.I),
    re.compile(r"\bJavaScript\b", re.I),
    re.compile(r"\bReact\b", re.I),
    re.compile(r"\b[eE]-?mail\b"),
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
]


def mask_entities(text: str) -> str:
    text = EMAIL_RE.sub(" [EMAIL] ", text)
    text = URL_RE.sub(" [URL] ", text)
    text = HANDLE_RE.sub(" [HANDLE] ", text)
    return text


def preprocess(text: str) -> str:
    text = normalize_unicode(text)
    text = mask_entities(text)
    return text


def make_adversarial_training_view(text: str, label: int) -> str:
    text = normalize_unicode(text)
    if not text:
        return text

    view = text
    if random.random() < 0.75:
        view = textbugger_augment(view, severity=0.12 if label == 0 else 0.22)
    if random.random() < 0.35:
        view = inject_punctuations(view, prob=0.03)
    if random.random() < 0.25:
        view = insert_homoglyphs(view, prob=0.08)
    if random.random() < 0.20:
        view = synonym_augment_sentence(view, n=1)[0]

    return preprocess(view)


def _binary_label_from_row(row: dict, columns: List[str]) -> int:
    if "label" in columns:
        try:
            return int(float(row.get("label", 0)) >= 0.5)
        except Exception:
            pass

    scores = []
    for col in REALWORLD_TOXIC_COLUMNS:
        if col in columns:
            try:
                scores.append(float(row.get(col, 0) or 0))
            except Exception:
                scores.append(0.0)

    if scores:
        return int(max(scores) >= 0.5)

    return 0


def _text_column(columns: List[str]) -> Optional[str]:
    for col in ("comment_text", "text", "comment", "content", "human_turn"):
        if col in columns:
            return col
    return columns[0] if columns else None


def _balanced_rows_from_dataset(ds, per_class: int, seed: int) -> List[Tuple[str, int]]:
    columns = list(ds.column_names)
    text_col = _text_column(columns)
    if not text_col:
        return []

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    positives, negatives = [], []

    for idx in indices:
        row = ds[idx]
        text = normalize_unicode(row.get(text_col, "")).strip()
        if len(text) < 5:
            continue

        label = _binary_label_from_row(row, columns)
        item = (text[:1024], label)

        if label == 1 and len(positives) < per_class:
            positives.append(item)
        elif label == 0 and len(negatives) < per_class:
            negatives.append(item)

        if len(positives) >= per_class and len(negatives) >= per_class:
            break

    return positives + negatives


def load_realworld_training_data(
    per_source: int = 3000,
    seed: int = SEED,
) -> Tuple[List[str], List[int]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"[WARN] datasets unavailable; using synthetic data only: {exc}")
        return [], []

    sources = [
        ("Jigsaw", "anitamaxvim/jigsaw-toxic-comments", None, "train"),
        ("Jigsaw mirror", "thesofakillers/jigsaw-toxic-comment-classification-challenge", None, "train"),
        ("Civil/SetFit", "SetFit/toxic_conversations", None, "train"),
        ("ToxicChat", "lmsys/toxic-chat", "toxicchat0124", "train"),
    ]

    texts, labels = [], []
    loaded_families = set()

    for name, repo, config, split in sources:
        family = "jigsaw" if "Jigsaw" in name else ("civil" if "Civil" in name else "toxicchat")
        if family in loaded_families:
            continue

        try:
            print(f"  Loading real-world training data: {name} ({repo})")
            if config:
                ds = load_dataset(repo, config, split=split, trust_remote_code=False)
            else:
                ds = load_dataset(repo, split=split, trust_remote_code=False)

            rows = _balanced_rows_from_dataset(
                ds,
                per_class=max(100, per_source // 2),
                seed=seed + len(loaded_families),
            )

            if rows:
                loaded_families.add(family)
                t, y = zip(*rows)
                texts.extend(t)
                labels.extend(y)
                print(f"    added {len(rows)} rows ({sum(y)} toxic, {len(y) - sum(y)} safe)")
        except Exception as exc:
            print(f"    skipped {name}: {exc}")

    return texts, labels


def stratified_split(
    texts: List[str],
    labels: List[int],
    val_ratio: float = 0.15,
    seed: int = SEED,
):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}

    for text, label in zip(texts, labels):
        by_label[int(label)].append((text, int(label)))

    train, val = [], []

    for rows in by_label.values():
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * val_ratio)) if len(rows) > 1 else 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    tr_texts, tr_labels = zip(*train) if train else ([], [])
    va_texts, va_labels = zip(*val) if val else ([], [])

    return list(tr_texts), list(va_texts), list(tr_labels), list(va_labels)


def text_fingerprint(text: str) -> str:
    norm = preprocess(normalize_unicode(text)).strip().lower()
    norm = re.sub(r"\s+", " ", norm)
    return hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()


def save_training_fingerprints(
    output_dir: str,
    train_texts: List[str],
    val_texts: List[str],
) -> None:
    payload = {
        "train": sorted({text_fingerprint(t) for t in train_texts}),
        "validation": sorted({text_fingerprint(t) for t in val_texts}),
    }
    payload["counts"] = {
        "train": len(payload["train"]),
        "validation": len(payload["validation"]),
    }

    Path(output_dir, "training_fingerprints.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def choose_threshold(
    labels: List[int],
    probs: List[float],
    min_recall: float = 0.86,
) -> Tuple[float, float]:
    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)

    if len(set(labels_arr.tolist())) < 2:
        return 0.55, 0.0

    best_th, best_score = 0.55, -1.0

    for th in np.linspace(0.20, 0.85, 132):
        preds = (probs_arr >= th).astype(int)
        tp = int(((preds == 1) & (labels_arr == 1)).sum())
        fp = int(((preds == 1) & (labels_arr == 0)).sum())
        fn = int(((preds == 0) & (labels_arr == 1)).sum())

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        score = f1 + (0.04 if rec >= min_recall else -0.08 * (min_recall - rec))

        if score > best_score:
            best_score, best_th = score, float(th)

    return best_th, best_score


EMAIL_EXAMPLES = [
    "alice@example.com", "nishanth@gmail.com", "first.last@my-college.edu",
    "name+tag@domain.co", "user_name@site.org",
]

URL_EXAMPLES = [
    "https://example.com", "http://my-site.org/docs", "www.domain.co/about",
    "app.example.io", "api.v1.service.local",
]

HANDLES = ["@alice", "@bob_the_builder", "@dev.ops", "@c_plus_plus", "@node.js"]

SAFE_WITH_SYMBOLS = [
    "be@utiful", "sm!le", "co-operate", "re:make", "a-b testing",
    "C++", "Node.js", "e-mail", "pi@no", "c# sharp", "pass-word",
    "cl@ss", "assess", "assistant", "@home", "re*act",
]

SAMPLE_SAFE_SENTENCES = [
    "I had a great day at the park.",
    "Please submit the report by Monday.",
    "Let's meet for coffee tomorrow.",
    "The documentation looks good.",
    "She's an excellent engineer.",
    "What's the weather like today?",
    "The project deadline is next Friday.",
    "Can you review this pull request?",
    "I really enjoyed the presentation.",
    "The new feature works perfectly.",
    "Thanks for your help with this!",
    "The algorithm converges in O(n log n).",
]

SAMPLE_INAPPROPRIATE_TEMPLATES = [
    "You are such a {}.", "What a {} move.", "I can't believe that {} did this.",
    "This is {} behavior.", "Go be a {}.", "Stop being a {}.",
    "That {} deserves nothing.", "Such a {} person.",
    "Everyone thinks you are a {}.", "Just a {} as always.",
]

SAMPLE_SEMANTIC_TOXIC_TEMPLATES = [
    "You are so {}.", "What a {} thing to say.", "That was {} of you.",
    "You {} excuse for a human.", "Absolutely {} behavior.",
]

BENIGN_SIMILAR = [
    "duck", "deck", "luck", "shift", "shelf", "pitch", "brick",
    "click", "stick", "trick", "class", "glass", "brass", "mass", "pass",
]

BENIGN_DOT_WORDS = [
    "nice", "neat", "cool", "good", "fine", "best", "kind", "care",
    "life", "love", "hope", "pure", "calm", "safe", "true", "fair",
    "wise", "bold", "dear", "glad", "free", "real", "soft", "warm",
]


def dotify(word: str) -> str:
    return ".".join(word)


def build_hard_negatives(n_repeat: int = 300):
    texts, labels = [], []
    pools = EMAIL_EXAMPLES + URL_EXAMPLES + HANDLES + SAFE_WITH_SYMBOLS

    templates = [
        "Contact me at {}.", "My handle is {}.", "Docs are here: {}",
        "We use {} in production.", "That looks {} to me.",
        "Please email {} for help.", "The tool is called {}.",
    ]

    for _ in range(n_repeat):
        choice = random.choice(pools)
        sent = random.choice(templates).format(choice) if random.random() < 0.6 else choice
        texts.append(sent)
        labels.append(0)

    dot_templates = [
        "{} meeting you.", "That was {} of them.", "What a {} day!",
        "Have a {} time.", "You are so {}.", "Everything feels {}.",
        "{} to hear that.", "Things are going {}.", "Stay {} out there.",
    ]

    for word in BENIGN_DOT_WORDS:
        for _ in range(6):
            dotted = dotify(word)
            if random.random() < 0.4:
                dotted = dotted[0].upper() + dotted[1:]

            texts.append(random.choice(dot_templates).format(dotted))
            labels.append(0)

            texts.append(dotted)
            labels.append(0)

    return texts, labels


def make_synthetic_dataset(
    n_positive_per_word: int = 300,
    n_negative: int = 3000,
    hard_negative_extra: int = 800,
    n_semantic_toxic: int = 500,
) -> Tuple[List[str], List[int]]:
    texts, labels = [], []

    for bad_word in BASE_PROFANITY:
        obfs = generate_obfuscations(bad_word, n=max(8, n_positive_per_word // 30))
        obfs.extend(generate_dot_variants(bad_word))

        hg_variant = insert_homoglyphs(bad_word, prob=0.6)
        if hg_variant != bad_word:
            obfs.append(hg_variant)

        texts.append(bad_word)
        labels.append(1)
        texts.append(bad_word.upper())
        labels.append(1)

        for _ in range(n_positive_per_word):
            template = random.choice(SAMPLE_INAPPROPRIATE_TEMPLATES)
            word_variant = random.choice(obfs)
            if random.random() < 0.15:
                word_variant = word_variant.upper()

            texts.append(template.format(word_variant))
            labels.append(1)

    for _ in range(n_semantic_toxic):
        seed = random.choice(SEMANTIC_TOXIC_SEEDS)
        template = random.choice(SAMPLE_SEMANTIC_TOXIC_TEMPLATES)
        sent = template.format(seed)

        for aug in synonym_augment_sentence(sent, n=2):
            texts.append(aug)
            labels.append(1)

        texts.append(sent)
        labels.append(1)

    for _ in range(n_negative):
        if random.random() < 0.6:
            sent = random.choice(SAMPLE_SAFE_SENTENCES)
        else:
            w = random.choice(BENIGN_SIMILAR)
            sent = random.choice([
                "That was a {}.", "I love my {}.", "The {} is broken.",
                "We need a {} here.", "Look at that {}.",
            ]).format(w)

        if random.random() < 0.3:
            sent = synonym_augment_sentence(sent, n=1)[0]

        texts.append(sent)
        labels.append(0)

    hn_texts, hn_labels = build_hard_negatives(hard_negative_extra)
    texts.extend(hn_texts)
    labels.extend(hn_labels)

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    return list(texts), list(labels)


CHAR_VOCAB = (
    " abcdefghijklmnopqrstuvwxyz0123456789"
    "!@#$%^&*()-_=+[]{}|;':\",./<>?\\`~"
)
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB) + 1
MAX_CHAR_LEN = 256


def text_to_char_ids(text: str, max_len: int = MAX_CHAR_LEN) -> List[int]:
    text = normalize_unicode(text).lower()[:max_len]
    ids = [CHAR2IDX.get(c, 0) for c in text]
    return ids + [0] * (max_len - len(ids))


class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int = CHAR_VOCAB_SIZE,
        emb_dim: int = 32,
        num_filters: int = 128,
        kernel_sizes: Tuple[int, ...] = (2, 3, 4, 5),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = num_filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))
            pooled.append(c.max(dim=2).values)
        return self.dropout(torch.cat(pooled, dim=1))


class DualEncoderDetector(nn.Module):
    def __init__(
        self,
        bert_hidden: int = 768,
        mlp_hidden: int = 256,
        num_labels: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.char_cnn = CharCNN(num_filters=128, kernel_sizes=(2, 3, 4, 5))

        fusion_in = bert_hidden + self.char_cnn.out_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        char_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :]
        char_emb = self.char_cnn(char_ids)
        fused = torch.cat([cls_emb, char_emb], dim=1)
        return self.classifier(fused)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits,
                self.smoothing / (num_classes - 1),
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        p_t = (probs * smooth_targets).sum(dim=-1)
        focal_weight = (1 - p_t) ** self.gamma

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        return (focal_weight * loss).mean()


class DualEncoderDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: DistilBertTokenizerFast,
        max_length: int = 192,
        token_dropout: float = 0.1,
        consistency_texts: Optional[List[str]] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_dropout = token_dropout
        self.is_train = token_dropout > 0
        self.mask_token_id = tokenizer.mask_token_id or tokenizer.pad_token_id or 0
        self.special_ids = {
            x for x in (
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            )
            if x is not None
        }

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        self.char_ids = [text_to_char_ids(t) for t in texts]

        self.consistency_texts = consistency_texts
        if consistency_texts is not None:
            self.view_encodings = tokenizer(
                consistency_texts,
                truncation=True,
                padding=False,
                max_length=max_length,
            )
            self.view_char_ids = [text_to_char_ids(t) for t in consistency_texts]
        else:
            self.view_encodings = None
            self.view_char_ids = None

    def __len__(self):
        return len(self.labels)

    def _maybe_token_dropout(self, ids: torch.Tensor) -> torch.Tensor:
        if not self.is_train or self.token_dropout <= 0:
            return ids

        ids = ids.clone()
        drop = torch.bernoulli(torch.full(ids.shape, self.token_dropout)).bool()
        special = torch.zeros_like(drop)

        for sid in self.special_ids:
            special |= ids.eq(sid)

        ids[drop & ~special] = self.mask_token_id
        return ids

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["input_ids"] = self._maybe_token_dropout(item["input_ids"])
        item["char_ids"] = torch.tensor(self.char_ids[idx], dtype=torch.long)
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.view_encodings is not None:
            view = {
                f"view_{k}": torch.tensor(v[idx])
                for k, v in self.view_encodings.items()
            }
            view["view_input_ids"] = self._maybe_token_dropout(view["view_input_ids"])
            view["view_char_ids"] = torch.tensor(
                self.view_char_ids[idx],
                dtype=torch.long,
            )
            item.update(view)

        return item


def dual_collate_fn(batch):
    input_ids = nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0,
    )
    char_ids = torch.stack([b["char_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "char_ids": char_ids,
        "labels": labels,
    }

    if "view_input_ids" in batch[0]:
        out["view_input_ids"] = nn.utils.rnn.pad_sequence(
            [b["view_input_ids"] for b in batch],
            batch_first=True,
            padding_value=0,
        )
        out["view_attention_mask"] = nn.utils.rnn.pad_sequence(
            [b["view_attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0,
        )
        out["view_char_ids"] = torch.stack([b["view_char_ids"] for b in batch])

    return out


def freelb_step(
    model: DualEncoderDetector,
    batch: dict,
    loss_fn: nn.Module,
    adv_steps: int = 3,
    adv_lr: float = 1e-2,
    adv_init_mag: float = 2e-2,
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    char_ids = batch["char_ids"]
    labels = batch["labels"]

    embedding_layer = model.bert.embeddings.word_embeddings

    with torch.no_grad():
        embeds_init = embedding_layer(input_ids)

    delta = torch.zeros_like(embeds_init).uniform_(-adv_init_mag, adv_init_mag)
    delta.requires_grad_(True)

    total_loss = torch.tensor(0.0, device=DEVICE)

    for _ in range(adv_steps):
        perturbed = embeds_init + delta

        bert_out = model.bert(
            inputs_embeds=perturbed,
            attention_mask=attention_mask,
        )
        cls_emb = bert_out.last_hidden_state[:, 0, :]
        char_emb = model.char_cnn(char_ids)
        logits = model.classifier(torch.cat([cls_emb, char_emb], dim=1))

        loss = loss_fn(logits, labels)
        (loss / adv_steps).backward()
        total_loss = total_loss + loss.detach()

        delta_grad = delta.grad.detach()
        delta = delta + adv_lr * delta_grad.sign()
        delta = torch.clamp(delta, -adv_init_mag, adv_init_mag).detach()
        delta.requires_grad_(True)

    return total_loss / adv_steps


def initialize_bert_from_classifier(
    model: DualEncoderDetector,
    classifier_dir: Optional[str],
) -> None:
    if not classifier_dir:
        return

    print(f"Initializing DistilBERT branch from baseline classifier: {classifier_dir}")
    seq_model = DistilBertForSequenceClassification.from_pretrained(classifier_dir)
    model.bert.load_state_dict(seq_model.distilbert.state_dict(), strict=True)
    del seq_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def symmetric_kl_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    log_p = F.log_softmax(logits_a / temperature, dim=-1)
    log_q = F.log_softmax(logits_b / temperature, dim=-1)
    p = log_p.exp()
    q = log_q.exp()

    kl_pq = F.kl_div(log_p, q, reduction="batchmean")
    kl_qp = F.kl_div(log_q, p, reduction="batchmean")

    return 0.5 * (kl_pq + kl_qp) * (temperature ** 2)


def supervised_consistency_loss(
    model: DualEncoderDetector,
    batch: dict,
    loss_fn: nn.Module,
    consistency_alpha: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    clean_logits = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["char_ids"],
    )
    clean_loss = loss_fn(clean_logits, batch["labels"])

    if "view_input_ids" not in batch:
        return clean_loss, clean_logits

    view_logits = model(
        batch["view_input_ids"],
        batch["view_attention_mask"],
        batch["view_char_ids"],
    )
    view_loss = loss_fn(view_logits, batch["labels"])
    consistency = symmetric_kl_loss(clean_logits, view_logits)

    loss = 0.5 * (clean_loss + view_loss) + consistency_alpha * consistency
    return loss, clean_logits


def train_model(
    output_dir: str,
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-5,
    adv_training: bool = True,
    adv_steps: int = 3,
    warmup_ratio: float = 0.1,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.08,
    include_realworld: bool = True,
    realworld_per_source: int = 3000,
    consistency_alpha: float = 0.25,
    max_length: int = 192,
    init_bert_from: Optional[str] = None,
    synthetic_scale: float = 0.50,
) -> str:
    print("Building synthetic robustness dataset...")

    scale = max(0.0, synthetic_scale)
    texts, labels = make_synthetic_dataset(
        n_positive_per_word=max(20, int(220 * scale)),
        n_negative=max(400, int(2600 * scale)),
        hard_negative_extra=max(150, int(900 * scale)),
        n_semantic_toxic=max(100, int(900 * scale)),
    )

    if include_realworld:
        print("Loading real-world toxicity data for domain adaptation...")
        rw_texts, rw_labels = load_realworld_training_data(
            per_source=realworld_per_source,
            seed=SEED,
        )

        if rw_texts:
            texts.extend(rw_texts)
            labels.extend(rw_labels)

            for t, y in zip(rw_texts, rw_labels):
                texts.append(textbugger_augment(t, severity=0.10 if y == 0 else 0.20))
                labels.append(y)

            print(f"  real-world + attacked rows added: {2 * len(rw_texts)}")
        else:
            print("  no real-world rows loaded; continuing with synthetic data")

    texts = [preprocess(t) for t in texts]
    labels = [int(y) for y in labels]

    train_texts, val_texts, train_labels, val_labels = stratified_split(
        texts,
        labels,
        val_ratio=0.15,
        seed=SEED,
    )

    train_views = [
        make_adversarial_training_view(t, y)
        for t, y in zip(train_texts, train_labels)
    ]

    print(
        f"Dataset: {len(train_texts)} train / {len(val_texts)} val | "
        f"train toxic={sum(train_labels)} safe={len(train_labels) - sum(train_labels)}"
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = DualEncoderDataset(
        train_texts,
        train_labels,
        tokenizer,
        max_length=max_length,
        token_dropout=0.08,
        consistency_texts=train_views,
    )
    val_dataset = DualEncoderDataset(
        val_texts,
        val_labels,
        tokenizer,
        max_length=max_length,
        token_dropout=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dual_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dual_collate_fn,
        num_workers=0,
    )

    model = DualEncoderDetector().to(DEVICE)
    initialize_bert_from_classifier(model, init_bert_from)

    loss_fn = FocalLoss(gamma=focal_gamma, smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    best_threshold = 0.55

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_training_fingerprints(output_dir, train_texts, val_texts)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()

            if adv_training:
                adv_loss = freelb_step(
                    model,
                    batch,
                    loss_fn,
                    adv_steps=adv_steps,
                    adv_lr=1e-2,
                    adv_init_mag=2e-2,
                )
                clean_loss, _ = supervised_consistency_loss(
                    model,
                    batch,
                    loss_fn,
                    consistency_alpha=consistency_alpha,
                )
                clean_loss.backward()
                combined_loss = (adv_loss + clean_loss.detach()) / 2
            else:
                clean_loss, _ = supervised_consistency_loss(
                    model,
                    batch,
                    loss_fn,
                    consistency_alpha=consistency_alpha,
                )
                clean_loss.backward()
                combined_loss = clean_loss.detach()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(combined_loss.item())

            if step % 50 == 0 or step == len(train_loader) - 1:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"step {step + 1}/{len(train_loader)} | "
                    f"avg_loss={total_loss / max(1, step + 1):.4f}",
                    flush=True,
                )

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["char_ids"],
                )
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(batch["labels"].cpu().numpy().tolist())

        epoch_threshold, _ = choose_threshold(all_labels, all_probs)
        all_preds = [int(p >= epoch_threshold) for p in all_probs]

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {acc:.4f} | Val F1: {f1:.4f} | th={epoch_threshold:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = epoch_threshold

            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            tokenizer.save_pretrained(output_dir)

            config = {
                "base_threshold": best_threshold,
                "max_length": max_length,
                "architecture": "distilbert_charcnn_act",
                "training": {
                    "include_realworld": include_realworld,
                    "realworld_per_source": realworld_per_source,
                    "consistency_alpha": consistency_alpha,
                    "adv_training": adv_training,
                    "adv_steps": adv_steps,
                    "init_bert_from": init_bert_from,
                    "synthetic_scale": synthetic_scale,
                },
            }

            Path(output_dir, "detector_config.json").write_text(
                json.dumps(config, indent=2),
                encoding="utf-8",
            )

            print(
                f"  -> Saved best model "
                f"(F1={best_f1:.4f}, th={best_threshold:.3f})"
            )

    print(
        f"\nTraining complete. "
        f"Best Val F1: {best_f1:.4f} | threshold={best_threshold:.3f}"
    )

    return output_dir


DELEET_MAP = str.maketrans({
    "@": "a", "4": "a", "8": "b", "(": "c", "3": "e",
    "1": "i", "!": "i", "0": "o", "$": "s", "5": "s",
    "7": "t", "+": "t",
})


def deleet(text: str) -> str:
    return text.translate(DELEET_MAP)


_PUNCT_SEPARATED_RE = re.compile(
    r"(?<!\w)([A-Za-z])(?:[.\-*_~]+[A-Za-z]){2,}(?!\w)"
)


def depunctuate(text: str) -> str:
    def _collapse(m: re.Match) -> str:
        return re.sub(r"[^A-Za-z]", "", m.group())

    return _PUNCT_SEPARATED_RE.sub(_collapse, text)


def collapse_repeats(text: str, max_repeat: int = 2) -> str:
    return re.sub(
        r"(.)\1{%d,}" % max_repeat,
        lambda m: m.group(1) * max_repeat,
        text,
    )


STRICT_TOXIC_ANCHORS = {
    "fuck", "fucker", "motherfucker", "motherfucking", "shit", "shitty",
    "bitch", "bitches", "bastard", "asshole", "dick", "cock", "cunt",
    "whore", "slut", "nigger", "nigga", "retard", "dumbass", "bullshit",
}

CONTEXTUAL_TOXIC_ANCHORS = set(SEMANTIC_TOXIC_SEEDS) | {
    "moron", "idiotic", "moronic", "imbecile", "worthless", "useless",
    "kill", "die", "trash", "garbage", "filth", "scum", "vermin",
}


def _anchor_variants(text: str) -> List[str]:
    norm = normalize_unicode(text).lower()
    return list(dict.fromkeys([
        norm,
        deleet(norm),
        depunctuate(norm),
        depunctuate(deleet(norm)),
        collapse_repeats(depunctuate(deleet(norm))),
    ]))


def toxic_rule_score(text: str) -> float:
    variants = _anchor_variants(text)

    for variant in variants:
        tokens = set(re.findall(r"[a-z]+", variant))

        if tokens & STRICT_TOXIC_ANCHORS:
            return 0.98

        contextual = tokens & CONTEXTUAL_TOXIC_ANCHORS
        if contextual:
            targeted = re.search(
                r"\b(you|u|he|she|they|them|those|these|all|go|your|their)\b",
                variant,
            )
            identity = any(term in tokens for term in IDENTITY_TERMS)
            imperative = re.search(r"\b(kill|die|go|send|deport|hate)\b", variant)

            if targeted or identity or imperative:
                return 0.90

    return 0.0


def is_whitelisted(text: str) -> bool:
    text = normalize_unicode(text).strip()

    if not text or toxic_rule_score(text) >= 0.90:
        return False

    masked = mask_entities(text).strip()
    entity_only = re.fullmatch(
        r"(?:\[EMAIL\]|\[URL\]|[\s,.;:()\-_/])+",
        masked,
    )

    if entity_only:
        return True

    for pat in SAFE_SYMBOL_PATTERNS:
        if pat.fullmatch(text):
            return True

    return False


class RobustDetector:
    def __init__(
        self,
        model_dir: str,
        device: str = None,
        base_threshold: Optional[float] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)

        config_path = self.model_dir / "detector_config.json"
        self.config = {}

        if config_path.exists():
            try:
                self.config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                self.config = {}

        self.max_length = int(self.config.get("max_length", 192))
        configured_threshold = float(self.config.get("base_threshold", 0.55))
        self.base_threshold = (
            configured_threshold if base_threshold is None else base_threshold
        )

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DualEncoderDetector()
        self.model.load_state_dict(
            torch.load(
                str(self.model_dir / "best_model.pt"),
                map_location=self.device,
                weights_only=True,
            )
        )
        self.model.to(self.device)
        self.model.eval()

    def _encode_batch(self, texts: List[str]) -> dict:
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        char_ids = torch.tensor(
            [text_to_char_ids(t) for t in texts],
            dtype=torch.long,
        )

        return {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
            "char_ids": char_ids.to(self.device),
        }

    def _score_batch(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            batch = self._encode_batch(texts)
            logits = self.model(**batch)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return probs

    def predict(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, float, str]]:
        th = threshold if threshold is not None else self.base_threshold
        texts = ["" if t is None else str(t) for t in texts]

        results = [None] * len(texts)
        to_model_idx = []
        rule_scores = {}

        for i, text in enumerate(texts):
            norm = normalize_unicode(text)

            if not norm.strip():
                results[i] = (0, 0.0, "empty")
                continue

            rule = toxic_rule_score(norm)
            rule_scores[i] = rule

            if rule >= 0.98:
                results[i] = (1, rule, "lexical_prior")
            elif is_whitelisted(norm):
                results[i] = (0, 0.0, "whitelist")
            else:
                to_model_idx.append(i)

        if not to_model_idx:
            return results

        view_texts = []

        for i in to_model_idx:
            norm = normalize_unicode(texts[i])
            lower = norm.lower()
            view_texts.append([
                preprocess(norm),
                preprocess(lower),
                preprocess(deleet(lower)),
                preprocess(deleet(norm)),
                preprocess(depunctuate(lower)),
                preprocess(collapse_repeats(depunctuate(deleet(lower)))),
            ])

        flat_texts = [v for views in view_texts for v in views]
        flat_probs = self._score_batch(flat_texts)

        n_views = 6

        for j, i in enumerate(to_model_idx):
            view_probs = flat_probs[j * n_views: j * n_views + n_views]

            model_prob = float(
                0.55 * np.max(view_probs) + 0.45 * np.mean(view_probs)
            )
            rule_prob = float(rule_scores.get(i, 0.0))
            agg_prob = 1.0 - (1.0 - model_prob) * (1.0 - rule_prob)

            text = texts[i]
            sym_density = sum(
                1 for c in text
                if not c.isalnum() and not c.isspace()
            ) / max(1, len(text))

            effective_th = (
                th + 0.04
                if sym_density > 0.22 and rule_prob < 0.90
                else th
            )

            pred = 1 if agg_prob >= effective_th else 0
            reason = "model_plus_prior" if rule_prob > 0 else "multi_view_vote"

            results[i] = (int(pred), agg_prob, reason)

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="robust_detector_model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--no_realworld", action="store_true")
    parser.add_argument("--realworld_per_source", type=int, default=3000)
    parser.add_argument("--consistency_alpha", type=float, default=0.25)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--init_bert_from", default=None)
    parser.add_argument("--synthetic_scale", type=float, default=0.50)
    
    args = parser.parse_args([
        '--output_dir', 'robust_detector_model',
        '--epochs', '3',
        '--batch_size', '16',
        '--realworld_per_source', '3000',
        '--consistency_alpha', '0.15'
    ])

    out = train_model(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        include_realworld=not args.no_realworld,
        realworld_per_source=args.realworld_per_source,
        consistency_alpha=args.consistency_alpha,
        max_length=args.max_length,
        init_bert_from=args.init_bert_from,
        synthetic_scale=args.synthetic_scale,
    )

    test_cases = [
        ("You are a f@ck", 1),
        ("What a d!ck move", 1),
        ("what the f.u.c.k", 1),
        ("sh1tty behavior", 1),
        ("\u0430sshole", 1),
        ("f\u03c5ck you", 1),
        ("\u0455hit happens", 1),
        ("You are so idiotic", 1),
        ("What a moronic thing to say", 1),
        ("That was pathetic of you", 1),
        ("She did a great job", 0),
        ("He is a duck", 0),
        ("nishanth@gmail.com", 0),
        ("She is be@utiful", 0),
        ("The C++ code is working", 0),
        ("N.i.c.e meeting you", 0),
        ("g.o.o.d morning everyone", 0),
        ("Have a g.r.e.a.t day", 0),
    ]

    print(f"\n{'Input':<40} {'Expected':>8} {'Pred':>6} {'Conf':>6} {'Reason'}")
    print("-" * 80)

    det = RobustDetector(out)

    for text, expected in test_cases:
        pred, conf, reason = det.predict([text])[0]
        flag = "OK" if pred == expected else "FAIL"
        print(f"{flag} {text:<38} {expected:>8} {pred:>6} {conf:>6.3f} {reason}")


