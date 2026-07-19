# Deploy Aestra (FastAPI + static UI)

The app is **not** a static site — GitHub Pages alone won’t work. You need a host that
runs Python (uvicorn).

## Option A — Render (recommended, free tier)

1. Push `fed-merge-finished` (or merge to a branch Render can see).
2. Go to [https://render.com](https://render.com) → **New** → **Web Service**.
3. Connect the GitHub repo (`Hamza-Ammar2/demo` or your fork).
4. Settings:
   - **Branch:** `fed-merge-finished`
   - **Runtime:** Python 3
   - **Build command (free-tier friendly — no torch):**  
     `pip install -r requirements.txt -e . && pip install openai`
   - **Start command:**  
     `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Add env vars in the dashboard (optional):
   - `OPENAI_API_KEY` — warmer doctor follow-ups
   - `CYCLEBENCH_USE_PFL=1` (uses GRU if torch is installed; else sklearn fallback)
   - `HF_TOKEN` / `HF_DATASET_REPO` — only if you want consented HF sync
6. Deploy → you get a URL like `https://aestra-xxxx.onrender.com`

**Torch / pFL on a paid plan:** append  
`&& pip install -r requirements-pfl.txt` to the build command (free tier often OOMs).

Or use the Blueprint: this repo’s `render.yaml` (“New Blueprint Instance”).

**Note:** Free tier spins down after idle; first request can take ~30–60s.

## Option B — Railway

1. [railway.app](https://railway.app) → New Project → Deploy from GitHub.
2. Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
3. Build: install deps as above (or use the `Dockerfile`).

## Option C — Docker locally / any VPS

```bash
docker build -t aestra .
docker run -p 8000:8000 -e OPENAI_API_KEY=... aestra
# open http://localhost:8000
```

## What gets served

| Path | Content |
|------|---------|
| `/` | Aestra home |
| `/analyse` | Feeling-off soft read |
| `/static/*` | CSS/JS/favicon |
| `/analyse/feeling-off` etc. | API |

Models and foundation JSON are copied into the image / available in the repo so the
demo works **without** mcPHASES/SWAN on the server. Restricted raw datasets stay local.

## Favicon

Pink circle: `web/favicon.svg` → `/static/favicon.svg`.
