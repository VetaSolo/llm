# LLM text pipeline

Учебный mini-product: один Python-пайплайн, который принимает сырой текст, извлекает смысл, классифицирует запрос, строит JSON по схеме и отвечает в стиле выбранной категории. Это не production-платформа и не RAG/агентная система.

Пайплайн:

```
текст
  → extract      (summary + 3 факта)
  → classify     (category, intent, sentiment)
  → structure    (сборка JSON в коде)
  → route        (dict в Python, не внутри промпта)
  → answer       (ответ в стиле категории)
  → self-check   (противоречия и потерянные детали)
  → revise       (если check не ок)
```

Категории: `support`, `feedback`, `complaint`, `sales`, `general_question`.

## Требования

- Python 3.11+
- Ключ OpenAI-совместимого API (удобно [Groq](https://console.groq.com))

## Установка

```powershell
cd path\to\llm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

На macOS/Linux: `source .venv/bin/activate`.

## API-ключ

В `.env` (файл в `.gitignore`, в git его нет) уже заданы Groq-ключ, модель и retry:

```
OPENAI_API_KEY=...
OPENAI_MODEL=llama-3.3-70b-versatile
OPENAI_BASE_URL=https://api.groq.com/openai/v1
LLM_MAX_RETRIES=3
LLM_RETRY_BASE=0.5
```

Ключ Groq начинается с `gsk_`. Для официального OpenAI уберите `OPENAI_BASE_URL` и поставьте, например, `OPENAI_MODEL=gpt-4o-mini`.

Не вставляйте ключ в чат и не коммитьте `.env`.

## Запуск

```powershell
python main.py --demo
```

Пять витринных сценариев (по одному на категорию). Результаты: `outputs/results.json`, лог: `outputs/pipeline.log`.

Свой текст:

```powershell
python main.py --text "Не могу войти после сброса пароля"
python main.py --file examples\04_billing_complaint.txt
python main.py
```

Без аргументов обрабатываются все файлы из `examples/*.txt`.

```powershell
python main.py --help
python main.py --list
```

## Пять демонстрационных сценариев

| # | Команда | Что показывает |
|---|---------|----------------|
| 1 | `python main.py --demo` | Полный цикл на 5 типах запросов |
| 2 | `python main.py --text "..."` | Произвольный ввод без правки кода |
| 3 | `python main.py --demo-routing` | Одна жалоба → три разных ответа (routing в коде) |
| 4 | `python main.py --demo-errors` | Сломанный JSON и схема, без API |
| 5 | `python main.py --demo-failures` | Текст вместо JSON, нет полей, падение API, fallback |

Сценарий 1 по файлам:

| Файл | Ожидаемая категория | Зачем |
|------|---------------------|--------|
| `examples/01_laptop_overheating.txt` | support | чеклист, без «обязательно рассмотрим» |
| `examples/02_app_feedback.txt` | feedback | благодарность, без даты релиза |
| `examples/03_rag_question.txt` | general_question | прямой ответ, не учебник |
| `examples/04_billing_complaint.txt` | complaint | извинение за конкретный вред + действие |
| `examples/06_sales_trial.txt` | sales | коротко: польза + CTA |

В `examples/` ещё 5 текстов (логин, доставка, цены, дайджест, черновик встречи) — они едут в `python main.py`.

## Пример входа и выхода

Вход (`examples/04_billing_complaint.txt`): двойное списание 15 марта, чат молчит 9 часов, требуют возврат.

Выход (сокращённо, полный шаблон — `examples/sample_output.json`):

```json
{
  "category": "complaint",
  "intent": "request_refund",
  "sentiment": "negative",
  "route": "complaint",
  "summary": "Двойная оплата подписки 15 марта.",
  "key_points": ["15 марта", "9 часов", "Роспотребнадзор"],
  "final_answer": "Извинение за двойное списание + конкретный возврат, без «напишите в поддержку»."
}
```

Если модель вернула мусор, пайплайн не падает: шаг помечается `fallback`, ответ всё равно есть (`STATUS: degraded`).

## Day 2: сравнение промптов

На одних и тех же 5 текстах сравнивали три формулировки. Считали не «красоту», а удержание формата (summary ≤ 2 предложения / 40 слов, ровно 3 key points, короткий ответ). Максимум 6 баллов на текст, итого /30.

| Вариант | Идея | Счёт |
|---|---|---|
| A `baseline_user_only` | Все правила в user-сообщении, почти без system | 23/30 |
| B `strict_system_constraints` | System = форматтер с жёсткими лимитами, user = сырой текст | **30/30** |
| C `warm_persona` | Сильная роль, слабый формат | ~0/30 |

Победил **B**. Лимиты реально держатся. C часто звучал человечнее, но писал `## SUMMARY` вместо `SUMMARY:` — парсер не вытаскивал поля. Для пайплайна это полный провал.

Сами тексты A/B/C оставлены закомментированными в `prompts.py`. Живой пайплайн их не вызывает: после Day 3 победитель переписан в JSON (`EXTRACT` / `CLASSIFY` / `ROUTES`).

## Структура

```
main.py          CLI
pipeline.py      цепочка шагов
llm_client.py    только вызов API + retry
prompts.py       system/user промпты и ROUTES
schemas.py       Pydantic-контракт
utils.py         fallback и лимиты длины
examples/        входы + sample_output.json
.env             ключ и настройки API (не коммитить)
```

## Конфигурация

| Переменная | Смысл |
|------------|--------|
| `OPENAI_API_KEY` | ключ |
| `OPENAI_MODEL` | имя модели |
| `OPENAI_BASE_URL` | совместимый endpoint (Groq, OpenRouter, Ollama) |
| `LLM_MAX_RETRIES` | попытки при сетевом сбое (по умолчанию 3) |
| `LLM_RETRY_BASE` | пауза первой повторной попытки в секундах |

## Что сознательно не входит

RAG, агенты, веб-UI, FastAPI, база знаний с ценами. Self-check ловит потерянные факты из письма, а не выдумывает неизвестную цену тарифа.
