# Aegis

Инструментарий для red-teaming и защиты LLM от prompt injection атак.

Цикл: **генерируем атаки → детектируем → усиливаем защиту → повторяем**.

## Структура проекта

```
├── collect_datasets.py    # Сборщик датасетов из 15 источников (691K примеров, 26 языков)
├── pyproject.toml         # Зависимости (uv)
└── docs/                  # Презентации
```

## Быстрый старт

```bash
# Собрать датасет с нуля
uv run collect_datasets.py

# С HuggingFace токеном (для gated датасетов)
uv run collect_datasets.py --hf-token YOUR_TOKEN

# Или загрузить готовый
python -c "from datasets import load_dataset; ds = load_dataset('Necent/llm-jailbreak-prompt-injection-dataset')"
```

### Параметры

| Флаг | Описание | Default |
|------|----------|---------|
| `--hf-token` | HuggingFace токен для gated датасетов | `$HF_TOKEN` |
| `--max-samples` | Лимит строк на большие датасеты | 50000 |
| `--polyglot-per-lang` | Лимит на язык для PolyglotToxicityPrompts | 2000 |
| `--skip-large` | Пропустить тяжёлые датасеты | false |
| `--output` | Папка для результатов | `data/` |

## Датасет

**[Necent/llm-jailbreak-prompt-injection-dataset](https://huggingface.co/datasets/Necent/llm-jailbreak-prompt-injection-dataset)** — 691K примеров, 26 языков

Схема:

| Колонка | Тип | Описание |
|---------|-----|----------|
| `prompt` | str | Текст промпта |
| `response` | str | Ответ модели (если есть) |
| `model_name` | str | Целевая модель |
| `prompt_type` | str | `jailbreak` / `prompt_injection` / `toxicity` / `linguistic` / `obfuscation` / `harmful_behavior` |
| `category` | str | Подкатегория из источника |
| `is_dangerous` | int | `1` = опасный, `0` = безопасный |
| `source` | str | Датасет-источник |
| `language` | str | ISO 639-1 код языка |

## Roadmap

- [x] Сбор датасета (691K, 15 источников, 26 языков)
- [x] Бенчмарк 3 моделей × 3 стратегии защиты
- [ ] Лёгкий классификатор — замена T-pro в model_guardrail
- [ ] Red-team LLM — генерация атак для расширения бенчмарка
- [ ] Defensive tokens — prefix tuning, защита на уровне весов

## Авторы

[Устинов Кирилл](https://github.com/umbilnm), [Борис Мацаков](https://github.com/Necentt)
