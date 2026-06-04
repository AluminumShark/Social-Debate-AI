# Social Debate AI - reproducible pipeline.
# Run `make help` to list targets. Requires Python 3.10+ and an LLM backend
# (a local/LAN Ollama by default; configure via .env -> LLM_BASE_URL / LLM_API_KEY).

PY ?= python

.PHONY: help setup models data index train eval demo run serve test lint \
        docker-app docker-train reproduce reproduce-full clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies and create .env from the template
	uv sync || pip install -e ".[dev]"
	@test -f .env || cp env.example .env
	@echo "Edit .env to point LLM_BASE_URL/LLM_API_KEY at your backend."

models:  ## Download pre-trained models + FAISS index from the GitHub release
	bash scripts/fetch_models.sh

data:  ## Download + clean the CMV corpus -> data/raw/{pairs,threads}.jsonl
	$(PY) scripts/prepare_cmv.py

index:  ## Build the FAISS RAG index (uses CMV pairs, or data/seed_evidence.jsonl)
	$(PY) scripts/build_rag_index.py

train:  ## Full training: data + FAISS index + graph GNN + PPO
	$(PY) train_all.py --all

eval:  ## Ablation A/B experiment -> docs/eval_results.md
	$(PY) scripts/eval_ab.py

demo:  ## Regenerate demo/sample_debate.json from a real run
	$(PY) scripts/make_demo.py

run:  ## Dev server (http://localhost:5000)
	$(PY) ui/app.py

serve:  ## Production server (gunicorn + threaded workers)
	gunicorn -k gthread -w 2 --threads 8 -t 0 -b 0.0.0.0:5000 wsgi:app

test:  ## Run the test suite
	USE_LLM_JUDGE=false $(PY) -m pytest tests/ -q

lint:  ## Lint with ruff
	$(PY) -m ruff check .

docker-app:  ## Build + run the web app in Docker
	docker compose -f docker/docker-compose.app.yml up -d --build

docker-train:  ## Run the GPU training pipeline in Docker
	docker compose -f docker/docker-compose.train.yml run --rm train

reproduce: setup index eval  ## Fork-friendly: deps -> RAG index (seed if no CMV) -> ablation (no GPU needed)
	@echo "Light reproduce complete. See docs/eval_results.md"

reproduce-full: setup data train eval  ## Full: download CMV + train GNN/RL + ablation (GPU recommended)
	@echo "Full reproduce complete. See docs/eval_results.md and data/models/"

clean:  ## Remove generated data/models/index (keeps seed_evidence)
	rm -rf data/models data/rag/faiss data/raw/pairs.jsonl data/raw/threads.jsonl data/debates.db
	@echo "Cleaned generated artifacts."
