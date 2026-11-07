# SVM Visualizer 


## Endpoints
https://svm-visualizer.fly.dev/
https://svm-visualizer.fly.dev/docs
https://svm-visualizer.fly.dev/redoc
https://fly-metrics.net/d/fly-logs/fly-logs?orgId=936389&var-app=svm-visualizer

## Development
Python 3.14 (`.python-version`). Direct dependencies live in `requirements.in` and `requirements-dev.in`; the hashed
`.txt` files are compiled from them with the `uv pip compile` command at the top of each file.

```bash
uv venv --python 3.14 venv
uv pip install --python venv --require-hashes -r requirements.txt -r requirements-dev.txt
venv/bin/python -m playwright install chromium
venv/bin/uvicorn src.app.main:app --reload          # uses ./app.db and ./error.log in the current directory
venv/bin/python -m pytest tests                     # starts the app itself from a temp copy of app.db
SVM_BASE_URL=https://svm-visualizer.fly.dev venv/bin/python -m pytest tests   # read-only against a running site
```
