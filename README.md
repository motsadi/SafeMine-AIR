# SafeMine AIR™ (Streamlit Demo App)

SafeMine AIR™ is a prototype Streamlit application that predicts safe re-entry times in underground mines after blasting by modeling toxic gas decay and ventilation performance using a **hybrid approach**:
- A physics baseline (dilution / clearance using an exponential decay model)
- A machine learning correction factor learned from historical data (synthetic data in this demo)

## Quick start (local)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Notes
- This demo generates **synthetic data** (no real mine data).
- The ML model learns a correction factor `alpha` such that:

`C(t) = C0 * exp(-(Q/V) * alpha * t)`

where `Q` is airflow rate (m³/s) and `V` is the effective volume (m³).

## Folder structure
- `app.py` – Streamlit app
- `requirements.txt` – dependencies
- `assets/` – branding images (placeholder)
- `data/` – generated synthetic dataset and trained model artifacts (created at runtime)

