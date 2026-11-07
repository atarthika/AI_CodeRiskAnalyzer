# AI-Powered Software Vulnerability & Code Risk Analyzer

Scans source code (C/C++, Python, SQL) for security problems and gives every project and file a 0-100 risk score.
It combines a **rule engine** (certain findings) with a trained **AI model** (suspicious areas), and explains each issue in simple words with a safer way to write the code.

## 1. Setup (one time)

Requires **Python 3.9+**.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 2. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser, paste code or upload files / a folder / a .zip, and click **Scan**.

## 3. Or use the command line

```bash
python Analyzer.py scan sample_project
python Analyzer.py scan path/to/your/code --json report.json --min-severity medium
```

Add `--no-ml` to use only the rule engine.

## 4. Train your own AI model (optional)

The `models/` folder already contains trained models. To retrain with your own dataset:

```bash
python Analyzer.py train dataset.csv --langs cpp
python Analyzer.py trainall --cpp path/to/cpp_data --sql path/to/sql_data
```

Datasets are not stored in this repo because of their size. Download them from Kaggle and point the commands above at them.

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Flask web server |
| `Analyzer.py` | Rule engine, AI model, training and command line |
| `static/index.html` | Web interface |
| `models/` | Trained AI models |
| `sample_project/` | Small project to test the scanner |
| `requirements.txt` | Python dependencies |

> Results are hints, not guarantees. Always review findings manually.
