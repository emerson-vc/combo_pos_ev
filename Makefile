PY := .venv/bin/python

.PHONY: setup ingest ingest-mlb ingest-mlb-lines ingest-nfl discover run test backtest

setup:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -r requirements.txt

ingest: ingest-nfl ingest-mlb ingest-mlb-lines

ingest-nfl:
	$(PY) -m app.ingest.nfl

ingest-mlb:
	$(PY) -m app.ingest.mlb

ingest-mlb-lines:
	$(PY) -m app.ingest.mlb_lines

discover:
	$(PY) -m app.kalshi.discovery

backtest:
	$(PY) -m app.model.backtest

run:
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

test:
	$(PY) -m pytest -q
