#!/usr/bin/env python3
"""
AI Safety Dataset Collector

Collects and unifies multiple AI safety/red-teaming datasets into a single
standardized format for training safety classifiers and evaluating LLM robustness.

Usage:
    uv run collect_datasets.py [--hf-token YOUR_TOKEN] [--max-samples N] [--output DIR]
"""

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Unified schema
COLUMNS = [
    "prompt",       # The attack/test prompt text
    "response",     # Model response if available
    "model_name",   # Target model if specified
    "prompt_type",  # jailbreak | prompt_injection | obfuscation | linguistic | toxicity | harmful_behavior
    "category",     # Specific sub-category from source
    "is_dangerous", # 1 = dangerous/harmful, 0 = safe/benign
    "source",       # Dataset name
    "language",     # ISO 639-1 language code
]


def empty_df():
    return pd.DataFrame(columns=COLUMNS)


LANG_NORMALIZE = {
    "english": "en", "chinese": "zh", "malay": "ms", "tamil": "ta",
    "singlish": "en", "arabic": "ar", "korean": "ko", "thai": "th",
    "bengali": "bn", "swahili": "sw", "javanese": "jv", "italian": "it",
    "vietnamese": "vi", "russian": "ru", "czech": "cs", "hungarian": "hu",
    "serbian": "sr", "spanish": "es", "french": "fr", "hindi": "hi",
    "indonesian": "id", "japanese": "ja", "dutch": "nl", "polish": "pl",
    "portuguese": "pt", "swedish": "sv", "german": "de",
}


def normalize_lang(lang):
    if not lang:
        return "en"
    lang = str(lang).strip().lower()
    return LANG_NORMALIZE.get(lang, lang[:2] if len(lang) > 2 else lang)


def make_row(prompt, response="", model_name="", prompt_type="", category="",
             is_dangerous=1, source="", language="en"):
    return {
        "prompt": str(prompt).strip() if prompt else "",
        "response": str(response).strip() if response else "",
        "model_name": str(model_name).strip() if model_name else "",
        "prompt_type": prompt_type,
        "category": category,
        "is_dangerous": int(is_dangerous),
        "source": source,
        "language": normalize_lang(language),
    }


# ─────────────────────────────────────────────────
# 1. JailbreakBench (JBB-Behaviors)
# ─────────────────────────────────────────────────
def collect_jbb_behaviors():
    """JailbreakBench/JBB-Behaviors: 100 harmful + 100 benign behaviors."""
    from datasets import load_dataset
    print("\n📥 Collecting JBB-Behaviors...")
    rows = []

    # Harmful split
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        for item in ds:
            rows.append(make_row(
                prompt=item.get("Goal", item.get("goal", "")),
                prompt_type="jailbreak",
                category=item.get("Category", item.get("category", "")),
                is_dangerous=1,
                source="JBB-Behaviors",
            ))
    except Exception as e:
        print(f"  ⚠️ Failed to load harmful split: {e}")

    # Benign split
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
        for item in ds:
            rows.append(make_row(
                prompt=item.get("Goal", item.get("goal", "")),
                prompt_type="jailbreak",
                category=item.get("Category", item.get("category", "")),
                is_dangerous=0,
                source="JBB-Behaviors",
            ))
    except Exception as e:
        print(f"  ⚠️ Failed to load benign split: {e}")

    print(f"  ✅ JBB-Behaviors: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 2. HarmBench (gated)
# ─────────────────────────────────────────────────
def collect_harmbench(token=None):
    """HarmBench: ~320 harmful behaviors. Try HF first, fallback to GitHub."""
    from datasets import load_dataset
    print("\n📥 Collecting HarmBench...")
    rows = []

    # Try HF gated dataset first
    if token:
        try:
            ds = load_dataset("walledai/HarmBench", split="train", token=token)
            for item in ds:
                rows.append(make_row(
                    prompt=item.get("prompt", ""),
                    prompt_type="jailbreak",
                    category="harmful_behavior",
                    is_dangerous=1,
                    source="HarmBench",
                ))
            if rows:
                print(f"  ✅ HarmBench (HF): {len(rows)} rows")
                return pd.DataFrame(rows, columns=COLUMNS)
        except Exception as e:
            print(f"  ⚠️ HF failed: {e}")

    # Fallback: GitHub CSV
    print("  🔄 Trying GitHub fallback...")
    url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    try:
        import io
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            for _, item in df.iterrows():
                behavior = item.get("Behavior", item.get("behavior", ""))
                if not behavior:
                    continue
                func_cat = item.get("FunctionalCategory", "")
                sem_cat = item.get("SemanticCategory", "")
                category = f"{func_cat}: {sem_cat}" if sem_cat else func_cat
                rows.append(make_row(
                    prompt=behavior,
                    prompt_type="jailbreak",
                    category=category if category else "harmful_behavior",
                    is_dangerous=1,
                    source="HarmBench",
                ))
    except Exception as e:
        print(f"  ⚠️ GitHub fallback failed: {e}")

    print(f"  ✅ HarmBench: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 3. AdvBench (gated)
# ─────────────────────────────────────────────────
def collect_advbench(token=None):
    """AdvBench: 520 harmful instructions. Try HF first, fallback to GitHub."""
    from datasets import load_dataset
    print("\n📥 Collecting AdvBench...")
    rows = []

    # Try HF first
    if token:
        try:
            ds = load_dataset("walledai/AdvBench", split="train", token=token)
            for item in ds:
                rows.append(make_row(
                    prompt=item.get("prompt", ""),
                    response=item.get("target", ""),
                    prompt_type="jailbreak",
                    category="adversarial",
                    is_dangerous=1,
                    source="AdvBench",
                ))
            if rows:
                print(f"  ✅ AdvBench (HF): {len(rows)} rows")
                return pd.DataFrame(rows, columns=COLUMNS)
        except Exception as e:
            print(f"  ⚠️ HF failed: {e}")

    # Fallback: GitHub CSV
    print("  🔄 Trying GitHub fallback...")
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    try:
        import io
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            for _, item in df.iterrows():
                goal = item.get("goal", "")
                target = item.get("target", "")
                if not goal:
                    continue
                rows.append(make_row(
                    prompt=goal,
                    response=target,
                    prompt_type="jailbreak",
                    category="adversarial",
                    is_dangerous=1,
                    source="AdvBench",
                ))
    except Exception as e:
        print(f"  ⚠️ GitHub fallback failed: {e}")

    print(f"  ✅ AdvBench: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 4. Do-Not-Answer
# ─────────────────────────────────────────────────
def collect_do_not_answer():
    """LibrAI/do-not-answer: 939 prompts with model responses and harm labels."""
    from datasets import load_dataset
    print("\n📥 Collecting Do-Not-Answer...")
    rows = []

    try:
        ds = load_dataset("LibrAI/do-not-answer", split="train")
        models = ["gpt4", "ChatGPT", "Claude", "llama2-7b-chat"]
        for item in ds:
            question = item.get("question", "")
            risk_area = item.get("risk_area", "")
            specific_harm = item.get("specific_harms", "")
            category_str = f"{risk_area}: {specific_harm}" if specific_harm else risk_area

            # Add the base prompt
            rows.append(make_row(
                prompt=question,
                prompt_type="harmful_behavior",
                category=category_str,
                is_dangerous=1,
                source="Do-Not-Answer",
            ))

            # Add model responses if available
            for model in models:
                resp_key = f"{model}_response"
                harm_key = f"{model}_harmful"
                resp = item.get(resp_key, "")
                harmful = item.get(harm_key)
                if resp and resp.strip():
                    rows.append(make_row(
                        prompt=question,
                        response=resp,
                        model_name=model,
                        prompt_type="harmful_behavior",
                        category=category_str,
                        is_dangerous=1,
                        source="Do-Not-Answer",
                    ))
    except Exception as e:
        print(f"  ⚠️ Failed: {e}")
        return empty_df()

    print(f"  ✅ Do-Not-Answer: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 5. TensorTrust
# ─────────────────────────────────────────────────
def collect_tensor_trust(max_samples=50000):
    """qxcv/tensor-trust: ~127K attack/defense examples from the game."""
    print("\n📥 Collecting TensorTrust...")
    rows = []

    # The HF dataset has mixed schemas, so download benchmark JSONL files directly
    hf_base = "https://huggingface.co/datasets/qxcv/tensor-trust/resolve/main/benchmarks"

    for bench_type in ["hijacking", "extraction"]:
        url = f"{hf_base}/{bench_type}-robustness/v1/{bench_type}_robustness_dataset.jsonl"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                for line in resp.text.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    attack = item.get("attack", item.get("attacker_input", ""))
                    if not attack:
                        continue
                    pre = item.get("pre_prompt", "")
                    post = item.get("post_prompt", "")
                    full_prompt = f"[System: {pre}]\n{attack}\n[System: {post}]" if pre else attack

                    rows.append(make_row(
                        prompt=full_prompt,
                        prompt_type="prompt_injection",
                        category=bench_type,
                        is_dangerous=1,
                        source="TensorTrust",
                    ))
                    if len(rows) >= max_samples:
                        break
        except Exception as e:
            print(f"  ⚠️ {bench_type} benchmark failed: {e}")

    # If GitHub didn't work, try HF with specific data files
    if len(rows) == 0:
        try:
            from datasets import load_dataset
            # Load each data file separately to avoid schema mismatch
            for data_file in [
                "data/hijacking_robustness_dataset-*.parquet",
                "data/extraction_robustness_dataset-*.parquet",
            ]:
                try:
                    ds = load_dataset("qxcv/tensor-trust", data_files=data_file,
                                      split="train", streaming=True)
                    for item in ds:
                        attack = item.get("attack", item.get("attacker_input", ""))
                        if not attack or not str(attack).strip():
                            continue
                        pre = item.get("pre_prompt", "")
                        post = item.get("post_prompt", "")
                        full_prompt = f"[System: {pre}]\n{attack}\n[System: {post}]" if pre else str(attack)
                        rows.append(make_row(
                            prompt=full_prompt,
                            response=item.get("llm_output", "") or "",
                            prompt_type="prompt_injection",
                            category="extraction" if "extraction" in data_file else "hijacking",
                            is_dangerous=1,
                            source="TensorTrust",
                        ))
                        if len(rows) >= max_samples:
                            break
                except Exception as e:
                    print(f"  ⚠️ HF data_file '{data_file}' failed: {e}")
        except Exception as e:
            print(f"  ⚠️ HF fallback failed: {e}")

    print(f"  ✅ TensorTrust: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 6. BIPIA (GitHub)
# ─────────────────────────────────────────────────
def collect_bipia():
    """microsoft/BIPIA: Indirect prompt injection attacks from GitHub."""
    print("\n📥 Collecting BIPIA...")
    rows = []
    base_url = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark"

    # Try to download attack files
    attack_types = ["text_attack_test", "code_attack_test", "text_attack_train", "code_attack_train"]
    for attack_type in attack_types:
        url = f"{base_url}/{attack_type}.json"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        attack_str = item.get("attack_str", item.get("attack", ""))
                        if attack_str:
                            rows.append(make_row(
                                prompt=attack_str,
                                prompt_type="prompt_injection",
                                category=f"indirect_{attack_type.split('_')[0]}",
                                is_dangerous=1,
                                source="BIPIA",
                            ))
                elif isinstance(data, dict):
                    for key, items in data.items():
                        if isinstance(items, list):
                            for item in items:
                                attack_str = item if isinstance(item, str) else item.get("attack_str", "")
                                if attack_str:
                                    rows.append(make_row(
                                        prompt=attack_str,
                                        prompt_type="prompt_injection",
                                        category=f"indirect_{attack_type.split('_')[0]}",
                                        is_dangerous=1,
                                        source="BIPIA",
                                    ))
            else:
                print(f"  ⚠️ {attack_type}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  ⚠️ {attack_type} failed: {e}")

    # Try task-specific test data
    tasks = ["email", "abstract", "table", "code", "qa"]
    for task in tasks:
        for split in ["test"]:
            url = f"{base_url}/{task}/{split}.jsonl"
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    for line in resp.text.strip().split("\n"):
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        context = item.get("context", item.get("email", item.get("content", "")))
                        question = item.get("question", item.get("instruction", ""))
                        if context or question:
                            prompt = f"{context}\n\n{question}" if context and question else (context or question)
                            rows.append(make_row(
                                prompt=prompt,
                                prompt_type="prompt_injection",
                                category=f"indirect_{task}",
                                is_dangerous=0,
                                source="BIPIA",
                            ))
            except Exception as e:
                print(f"  ⚠️ {task}/{split} failed: {e}")

    print(f"  ✅ BIPIA: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 7. LLMail-Inject
# ─────────────────────────────────────────────────
def collect_llmail_inject(max_samples=20000):
    """microsoft/llmail-inject-challenge: email-based prompt injection attacks."""
    from datasets import load_dataset
    print("\n📥 Collecting LLMail-Inject...")
    rows = []

    try:
        ds = load_dataset("microsoft/llmail-inject-challenge",
                          split="Phase1", streaming=True)
        for item in ds:
            body = item.get("body", "")
            subject = item.get("subject", "")
            prompt = f"Subject: {subject}\n\n{body}" if subject else body
            if not prompt.strip():
                continue

            rows.append(make_row(
                prompt=prompt,
                response=item.get("output", ""),
                prompt_type="prompt_injection",
                category="email_injection",
                is_dangerous=1,
                source="LLMail-Inject",
            ))
            if len(rows) >= max_samples:
                break
    except Exception as e:
        # Try default split
        try:
            ds = load_dataset("microsoft/llmail-inject-challenge",
                              split="train", streaming=True)
            for item in ds:
                body = item.get("body", "")
                subject = item.get("subject", "")
                prompt = f"Subject: {subject}\n\n{body}" if subject else body
                if not prompt.strip():
                    continue
                rows.append(make_row(
                    prompt=prompt,
                    response=item.get("output", ""),
                    prompt_type="prompt_injection",
                    category="email_injection",
                    is_dangerous=1,
                    source="LLMail-Inject",
                ))
                if len(rows) >= max_samples:
                    break
        except Exception as e2:
            print(f"  ⚠️ Failed: {e2}")
            return empty_df()

    print(f"  ✅ LLMail-Inject: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 8. SPML Dataset
# ─────────────────────────────────────────────────
def collect_spml():
    """reshabhs/SPML_Chatbot_Prompt_Injection: 16K system prompt + injection examples."""
    from datasets import load_dataset
    print("\n📥 Collecting SPML...")
    rows = []

    try:
        ds = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection", split="train")
        for item in ds:
            system_prompt = item.get("System Prompt", "")
            user_prompt = item.get("User Prompt", "")
            is_injection = item.get("Prompt injection", 0)

            prompt = f"[System: {system_prompt}]\n{user_prompt}" if system_prompt else user_prompt
            rows.append(make_row(
                prompt=prompt,
                prompt_type="prompt_injection",
                category=item.get("Source", "spml"),
                is_dangerous=int(is_injection),
                source="SPML",
            ))
    except Exception as e:
        print(f"  ⚠️ Failed: {e}")
        return empty_df()

    print(f"  ✅ SPML: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 9. WildGuardMix (gated)
# ─────────────────────────────────────────────────
def collect_wildguardmix(token=None):
    """allenai/wildguardmix: 92K examples with harm labels. Requires HF token."""
    from datasets import load_dataset
    print("\n📥 Collecting WildGuardMix...")
    if not token:
        print("  ⏭️ Skipped (requires HF token). Use --hf-token to provide one.")
        return empty_df()

    rows = []
    try:
        for split_name in ["train", "test"]:
            try:
                config = "WildGuardTrain" if split_name == "train" else "WildGuardTest"
                ds = load_dataset("allenai/wildguardmix", config, split=split_name,
                                  token=token)
                for item in ds:
                    prompt = item.get("prompt", "")
                    response = item.get("response", "")
                    harm_label = item.get("prompt_harm_label", "")
                    is_harmful = 1 if harm_label == "harmful" else 0

                    # Map to prompt_type
                    prompt_type = "jailbreak" if is_harmful else "harmful_behavior"

                    rows.append(make_row(
                        prompt=prompt,
                        response=response if response else "",
                        prompt_type=prompt_type,
                        category=harm_label,
                        is_dangerous=is_harmful,
                        source="WildGuardMix",
                    ))
            except Exception as e:
                print(f"  ⚠️ Split '{split_name}' failed: {e}")
    except Exception as e:
        print(f"  ⚠️ Failed: {e}")
        return empty_df()

    print(f"  ✅ WildGuardMix: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 10. RedBench
# ─────────────────────────────────────────────────
def collect_redbench():
    """knoveleng/redbench: 29K samples from 37 merged benchmarks."""
    from datasets import load_dataset
    print("\n📥 Collecting RedBench...")
    rows = []

    # RedBench has many configs. Try loading the main one first, then individual ones.
    configs = [
        "AdvBench", "CatQA", "CoCoNot", "CoNA", "CoSafe",
        "ControversialInstructions", "CyberattackAssistance", "DAN", "DeMET",
        "DiaSafety", "DoNotAnswer", "ForbiddenQuestions", "GEST", "GPTFuzzer",
        "GandalfIgnoreInstructions", "GandalfSummarization", "HarmBench",
        "HarmfulQ", "HarmfulQA", "JADE", "JBBBehaviours", "LatentJailbreak",
        "MaliciousInstruct", "MaliciousInstructions", "MedSafetyBench",
        "MoralExceptQA", "ORBench", "PhysicalSafetyInstructions", "QHarm",
        "SGBench", "SGXSTest", "SafeText", "StrongREJECT", "ToxiGen",
        "WMDP", "XSTest", "XSafety",
    ]

    for config in tqdm(configs, desc="  RedBench configs"):
        try:
            ds = load_dataset("knoveleng/redbench", config, split="test")
            for item in ds:
                prompt = item.get("prompt", "")
                if not prompt:
                    continue
                category = item.get("category", item.get("subtask", config))
                lang = item.get("language", "en")
                # Determine danger level from task/risk info
                risk = item.get("risk_response", item.get("answer", ""))
                is_dangerous = 1  # RedBench entries are mostly adversarial/harmful

                rows.append(make_row(
                    prompt=prompt,
                    prompt_type="jailbreak",
                    category=f"{config}: {category}" if category else config,
                    is_dangerous=is_dangerous,
                    source="RedBench",
                    language=lang if lang else "en",
                ))
        except Exception as e:
            # Some configs may not have a test split
            try:
                ds = load_dataset("knoveleng/redbench", config, split="train")
                for item in ds:
                    prompt = item.get("prompt", "")
                    if not prompt:
                        continue
                    category = item.get("category", config)
                    rows.append(make_row(
                        prompt=prompt,
                        prompt_type="jailbreak",
                        category=f"{config}: {category}" if category else config,
                        is_dangerous=1,
                        source="RedBench",
                        language=item.get("language", "en") or "en",
                    ))
            except Exception:
                print(f"  ⚠️ Config '{config}' failed: {e}")

    print(f"  ✅ RedBench: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 11. MultiJail
# ─────────────────────────────────────────────────
def collect_multijail():
    """DAMO-NLP-SG/MultiJail: 315 prompts in 10 languages."""
    from datasets import load_dataset
    print("\n📥 Collecting MultiJail...")
    rows = []

    lang_map = {
        "en": "en", "zh": "zh", "it": "it", "vi": "vi", "ar": "ar",
        "ko": "ko", "th": "th", "bn": "bn", "sw": "sw", "jv": "jv",
    }

    try:
        ds = load_dataset("DAMO-NLP-SG/MultiJail", split="train")
        for item in ds:
            tags = item.get("tags", "")
            source = item.get("source", "")
            for lang_col, lang_code in lang_map.items():
                text = item.get(lang_col, "")
                if text and text.strip():
                    rows.append(make_row(
                        prompt=text,
                        prompt_type="linguistic",
                        category=tags if tags else "multilingual_jailbreak",
                        is_dangerous=1,
                        source="MultiJail",
                        language=lang_code,
                    ))
    except Exception as e:
        print(f"  ⚠️ Failed: {e}")
        return empty_df()

    print(f"  ✅ MultiJail: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 12. PolyglotToxicityPrompts
# ─────────────────────────────────────────────────
def collect_polyglot_toxicity(max_samples_per_lang=2000):
    """swiss-ai/polyglotoxicityprompts: toxic prompts in 17 languages."""
    from datasets import load_dataset
    print("\n📥 Collecting PolyglotToxicityPrompts...")
    rows = []

    languages = ["ar", "cs", "de", "en", "es", "fr", "hi", "id", "it",
                 "ja", "ko", "nl", "pl", "pt", "ru", "sv", "zh"]

    # Collect from both ptp and wildchat configs, both small and full splits
    source_types = ["ptp", "wildchat"]

    for source_type in source_types:
        for lang in tqdm(languages, desc=f"  PolyglotToxicity {source_type}"):
            try:
                config = f"{source_type}-{lang}"
                # Try full split first, fall back to small
                for split_name in ["full", "small"]:
                    try:
                        ds = load_dataset("swiss-ai/polyglotoxicityprompts", config,
                                          split=split_name, streaming=True)
                        count = 0
                        for item in ds:
                            prompt = item.get("prompt", item.get("text", ""))
                            if not prompt or not prompt.strip():
                                continue
                            toxicity = item.get("toxicity", 0.0)
                            is_toxic = 1 if toxicity and toxicity > 0.5 else 0

                            rows.append(make_row(
                                prompt=prompt,
                                prompt_type="toxicity",
                                category=f"toxic_{source_type}_{lang}" if is_toxic else f"benign_{source_type}_{lang}",
                                is_dangerous=is_toxic,
                                source="PolyglotToxicityPrompts",
                                language=lang,
                            ))
                            count += 1
                            if count >= max_samples_per_lang:
                                break
                        break  # Successfully loaded, skip other splits
                    except Exception:
                        continue
            except Exception as e:
                print(f"  ⚠️ {source_type}-{lang} failed: {e}")

    print(f"  ✅ PolyglotToxicityPrompts: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 13. LinguaSafe
# ─────────────────────────────────────────────────
def collect_linguasafe():
    """zhiyuan-ning/linguasafe: 45K examples in 12 languages."""
    from datasets import load_dataset
    print("\n📥 Collecting LinguaSafe...")
    rows = []

    try:
        ds = load_dataset("zhiyuan-ning/linguasafe", split="train")
        for item in ds:
            prompt = item.get("prompt", item.get("text", item.get("content", "")))
            if not prompt:
                # Try to find the prompt field
                for key in item:
                    if isinstance(item[key], str) and len(item[key]) > 20:
                        prompt = item[key]
                        break
            if not prompt:
                continue

            lang = item.get("language", item.get("lang", "en"))
            severity = item.get("severity", item.get("level", ""))
            domain = item.get("domain", item.get("category", ""))
            # L0 = benign, L1-L3 = harmful
            is_dangerous = 0 if severity in ("L0", "0", 0) else 1

            rows.append(make_row(
                prompt=prompt,
                prompt_type="linguistic",
                category=f"{domain}" if domain else "linguasafe",
                is_dangerous=is_dangerous,
                source="LinguaSafe",
                language=str(lang).lower()[:2] if lang else "en",
            ))
    except Exception as e:
        print(f"  ⚠️ Failed: {e}")
        # Try alternative loading method
        try:
            print("  🔄 Trying alternative loading method...")
            ds = load_dataset("zhiyuan-ning/linguasafe", split="train",
                              trust_remote_code=True)
            for item in ds:
                first_str = ""
                for key, val in item.items():
                    if isinstance(val, str) and len(val) > 10:
                        first_str = val
                        break
                if first_str:
                    rows.append(make_row(
                        prompt=first_str,
                        prompt_type="linguistic",
                        category="linguasafe",
                        is_dangerous=1,
                        source="LinguaSafe",
                    ))
        except Exception as e2:
            print(f"  ⚠️ Alternative method also failed: {e2}")
            return empty_df()

    print(f"  ✅ LinguaSafe: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 14. RabakBench
# ─────────────────────────────────────────────────
def collect_rabakbench():
    """govtech/RabakBench: 528+ examples in Singlish, Chinese, Malay, Tamil."""
    from datasets import load_dataset
    print("\n📥 Collecting RabakBench...")
    rows = []

    lang_files = {
        "en": "rabakbench_en.csv",
        "zh": "rabakbench_zh.csv",
        "ms": "rabakbench_ms.csv",
        "ta": "rabakbench_ta.csv",
    }

    for lang_code, filename in lang_files.items():
        try:
            ds = load_dataset("govtech/RabakBench", data_files=filename, split="train")
            for item in ds:
                text = item.get("text", "")
                if not text:
                    continue
                is_unsafe = item.get("binary", 0)

                categories = []
                if item.get("hateful", 0) > 0:
                    categories.append("hateful")
                if item.get("insults", 0) > 0:
                    categories.append("insults")
                if item.get("sexual", 0) > 0:
                    categories.append("sexual")
                if item.get("physical_violence", 0) > 0:
                    categories.append("violence")
                if item.get("self_harm", 0) > 0:
                    categories.append("self_harm")
                if item.get("all_other_misconduct", 0) > 0:
                    categories.append("misconduct")

                category = ", ".join(categories) if categories else "safe"

                rows.append(make_row(
                    prompt=text,
                    prompt_type="toxicity",
                    category=category,
                    is_dangerous=int(is_unsafe),
                    source="RabakBench",
                    language=lang_code,
                ))
        except Exception as e:
            print(f"  ⚠️ {filename} failed: {e}")

    print(f"  ✅ RabakBench: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 15. ArtPrompt (GitHub)
# ─────────────────────────────────────────────────
def collect_artprompt():
    """uw-nsl/ArtPrompt: 50 harmful behaviors for ASCII art attacks."""
    print("\n📥 Collecting ArtPrompt...")
    rows = []
    url = "https://raw.githubusercontent.com/uw-nsl/ArtPrompt/main/dataset/harmful_behaviors_custom.csv"

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            import io
            df = pd.read_csv(io.StringIO(resp.text))
            for _, item in df.iterrows():
                goal = item.get("goal", "")
                if not goal:
                    continue
                rows.append(make_row(
                    prompt=goal,
                    prompt_type="obfuscation",
                    category=item.get("category", "ascii_art"),
                    is_dangerous=1,
                    source="ArtPrompt",
                ))
        else:
            print(f"  ⚠️ HTTP {resp.status_code}")
    except Exception as e:
        print(f"  ⚠️ Failed: {e}")

    print(f"  ✅ ArtPrompt: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 16. Mindgard Evaded Samples (gated)
# ─────────────────────────────────────────────────
def collect_mindgard(token=None):
    """Mindgard/evaded-prompt-injection-and-jailbreak-samples: 554 base + variants."""
    from datasets import load_dataset
    print("\n📥 Collecting Mindgard Evaded Samples...")
    if not token:
        print("  ⏭️ Skipped (requires HF token). Use --hf-token to provide one.")
        return empty_df()

    rows = []
    try:
        ds = load_dataset("Mindgard/evaded-prompt-injection-and-jailbreak-samples",
                          split="train", token=token)
        for item in ds:
            original = item.get("original_prompt", "")
            modified = item.get("modified_prompt", "")
            attack_name = item.get("attack_name", "")

            # Add original
            if original:
                rows.append(make_row(
                    prompt=original,
                    prompt_type="obfuscation",
                    category=f"original ({attack_name})",
                    is_dangerous=1,
                    source="Mindgard",
                ))
            # Add modified (obfuscated) variant
            if modified and modified != original:
                rows.append(make_row(
                    prompt=modified,
                    prompt_type="obfuscation",
                    category=attack_name,
                    is_dangerous=1,
                    source="Mindgard",
                ))
    except Exception as e:
        print(f"  ⚠️ Failed: {e}")
        return empty_df()

    print(f"  ✅ Mindgard: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# 17. SGToxicGuard (GitHub)
# ─────────────────────────────────────────────────
def collect_sgtoxicguard():
    """Social-AI-Studio/SGToxicGuard: toxicity detection for Singapore languages."""
    print("\n📥 Collecting SGToxicGuard...")
    rows = []
    base_url = "https://raw.githubusercontent.com/Social-AI-Studio/SGToxicGuard/main/dataset"

    lang_map = {"en": "en", "ss": "en", "zh": "zh", "ms": "ms", "ta": "ta"}

    for lang_code in lang_map:
        url = f"{base_url}/task1_{lang_code}.json"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"  ⚠️ task1_{lang_code}: HTTP {resp.status_code}")
                continue

            text = resp.text.strip()
            # Try JSONL first (one JSON object per line)
            items = []
            try:
                items = [json.loads(line) for line in text.split("\n") if line.strip()]
            except json.JSONDecodeError:
                # Try as single JSON array/object
                try:
                    data = json.loads(text)
                    items = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
                except json.JSONDecodeError as e2:
                    print(f"  ⚠️ task1_{lang_code} parse failed: {e2}")
                    continue

            for item in items:
                if isinstance(item, dict):
                    txt = item.get("text", item.get("sentence", item.get("content", "")))
                    label = item.get("label", item.get("toxicity", ""))
                    if txt:
                        is_toxic = 1 if label in ("toxic", "1", 1, True) else 0
                        rows.append(make_row(
                            prompt=txt,
                            prompt_type="toxicity",
                            category=f"sg_toxic_{lang_code}",
                            is_dangerous=is_toxic,
                            source="SGToxicGuard",
                            language=lang_map[lang_code],
                        ))
                elif isinstance(item, str) and item.strip():
                    rows.append(make_row(
                        prompt=item,
                        prompt_type="toxicity",
                        category=f"sg_toxic_{lang_code}",
                        is_dangerous=1,
                        source="SGToxicGuard",
                        language=lang_map[lang_code],
                    ))
        except Exception as e:
            print(f"  ⚠️ task1_{lang_code} failed: {e}")

    print(f"  ✅ SGToxicGuard: {len(rows)} rows")
    return pd.DataFrame(rows, columns=COLUMNS)


# ─────────────────────────────────────────────────
# Main collection pipeline
# ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Collect AI safety datasets")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""),
                        help="HuggingFace token for gated datasets")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Max samples per large dataset (default: 50000)")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--polyglot-per-lang", type=int, default=2000,
                        help="Max samples per language for PolyglotToxicityPrompts (default: 2000)")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip very large datasets (TensorTrust, LLMail, PolyglotToxicity)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or None

    print("=" * 60)
    print("🔬 AI Safety Dataset Collector")
    print("=" * 60)
    if token:
        print("✅ HuggingFace token provided — gated datasets will be included")
    else:
        print("⚠️  No HF token — gated datasets (HarmBench, AdvBench, WildGuardMix, Mindgard) will be skipped")
        print("   Set HF_TOKEN env var or use --hf-token to include them")
    print(f"📁 Output: {output_dir.resolve()}")
    print("=" * 60)

    # Collect all datasets
    collectors = [
        ("JBB-Behaviors", lambda: collect_jbb_behaviors()),
        ("HarmBench", lambda: collect_harmbench(token)),
        ("AdvBench", lambda: collect_advbench(token)),
        ("Do-Not-Answer", lambda: collect_do_not_answer()),
        ("SPML", lambda: collect_spml()),
        ("MultiJail", lambda: collect_multijail()),
        ("RabakBench", lambda: collect_rabakbench()),
        ("ArtPrompt", lambda: collect_artprompt()),
        ("SGToxicGuard", lambda: collect_sgtoxicguard()),
        ("Mindgard", lambda: collect_mindgard(token)),
        ("RedBench", lambda: collect_redbench()),
        ("BIPIA", lambda: collect_bipia()),
        ("LinguaSafe", lambda: collect_linguasafe()),
    ]

    if not args.skip_large:
        collectors.extend([
            ("TensorTrust", lambda: collect_tensor_trust(args.max_samples)),
            ("LLMail-Inject", lambda: collect_llmail_inject(args.max_samples)),
            ("PolyglotToxicityPrompts", lambda: collect_polyglot_toxicity(args.polyglot_per_lang)),
        ])
    else:
        print("\n⏭️ Skipping large datasets (--skip-large)")

    all_dfs = []
    failed = []

    for name, collector_fn in collectors:
        try:
            df = collector_fn()
            if len(df) > 0:
                all_dfs.append(df)
                # Also save individual dataset
                individual_path = output_dir / f"{name.lower().replace(' ', '_').replace('-', '_')}.parquet"
                df.to_parquet(individual_path, index=False)
        except Exception as e:
            print(f"\n❌ {name} completely failed: {e}")
            traceback.print_exc()
            failed.append(name)

    if not all_dfs:
        print("\n❌ No datasets collected! Check errors above.")
        sys.exit(1)

    # Combine all datasets
    print("\n" + "=" * 60)
    print("📊 Combining datasets...")
    combined = pd.concat(all_dfs, ignore_index=True)

    # Remove rows with empty prompts
    combined = combined[combined["prompt"].str.strip().str.len() > 0].reset_index(drop=True)

    # Remove exact duplicates
    before_dedup = len(combined)
    combined = combined.drop_duplicates(
        subset=["prompt", "response", "model_name", "source"],
        keep="first"
    ).reset_index(drop=True)
    after_dedup = len(combined)
    if before_dedup != after_dedup:
        print(f"  Removed {before_dedup - after_dedup} duplicates")

    # Save combined dataset
    csv_path = output_dir / "combined_dataset.csv"
    parquet_path = output_dir / "combined_dataset.parquet"

    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("📈 COLLECTION SUMMARY")
    print("=" * 60)
    print(f"\nTotal rows: {len(combined):,}")
    print(f"Dangerous prompts: {combined['is_dangerous'].sum():,}")
    print(f"Safe prompts: {(combined['is_dangerous'] == 0).sum():,}")

    print(f"\n{'Source':<30} {'Count':>8} {'Dangerous':>10} {'Safe':>6}")
    print("-" * 60)
    for source in sorted(combined["source"].unique()):
        subset = combined[combined["source"] == source]
        dangerous = subset["is_dangerous"].sum()
        safe = (subset["is_dangerous"] == 0).sum()
        print(f"{source:<30} {len(subset):>8,} {dangerous:>10,} {safe:>6,}")

    print(f"\n{'Prompt Type':<30} {'Count':>8}")
    print("-" * 40)
    for ptype in sorted(combined["prompt_type"].unique()):
        count = (combined["prompt_type"] == ptype).sum()
        print(f"{ptype:<30} {count:>8,}")

    print(f"\n{'Language':<10} {'Count':>8}")
    print("-" * 20)
    for lang in sorted(combined["language"].unique()):
        count = (combined["language"] == lang).sum()
        print(f"{lang:<10} {count:>8,}")

    print(f"\n📁 Saved to:")
    print(f"   CSV:     {csv_path.resolve()}")
    print(f"   Parquet: {parquet_path.resolve()}")
    print(f"   Individual datasets: {output_dir.resolve()}/*.parquet")

    if failed:
        print(f"\n⚠️ Failed datasets: {', '.join(failed)}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
