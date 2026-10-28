# Government vs Opposition

A Gradio app in which two language models debate the politics of a country you choose. One
model defends the government and the other plays the opposition. Each debate is limited to
10 rounds, where a round is one government turn followed by one opposition turn.

## How it works

1. You enter a country, an optional topic and the number of rounds.
2. The app downloads recent indicators for that country from the World Bank Open Data API
   (GDP growth, inflation, unemployment, debt, health and education spending, and others).
3. Both models receive the same data in their system prompts and must cite it. Each side
   argues from its own role and keeps its own conversation history.
4. Every reply is checked for numbers that are not in the data. If it contains any, the
   model is asked once to rewrite the reply using only the real figures.

The models run on Groq's free tier. The app asks Groq which models your key can use and
gives the government and the opposition two different ones. The model name is shown on
every message.

## Setup

You need Python 3.10 or newer and a free API key from https://console.groq.com/keys.

```
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your key after `GROQ_API_KEY=`. The `.env` file is
listed in `.gitignore`, so the key is never committed.

## Run

```
python app.py
```

Open the address it prints (http://127.0.0.1:7860), enter a country and press Start debate.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio interface |
| `debate.py` | Prompts, model selection, replies and the number check |
| `worldbank.py` | Country lookup and World Bank downloads |
| `style.css` | Page styling |

## Limitations

- The World Bank usually publishes figures with a delay of one or two years, and some
  countries lack some indicators. The statistics panel lists what was loaded.
- The check only covers numbers. The models can still make qualitative claims that are
  wrong, so treat the output as a debate and not as a source.
- The free Groq tier limits tokens per minute, so long debates may pause briefly.
