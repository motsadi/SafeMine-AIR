# SafeMine AIR™ (Streamlit App)

SafeMine AIR™ is a prototype Streamlit application that predicts safe re-entry times in underground mines after blasting by modelling toxic gas decay and ventilation performance using a **hybrid approach**:
- A physics baseline (dilution / clearance using an exponential decay model)
- A machine learning correction factor learned from historical data (simulated data from Ventsim)

## What is included

- A more polished visual dashboard for gas clearance forecasting
- Ventilation sufficiency alerts based on delivered airflow versus a target re-entry window
- Scenario testing to compare airflow, fan, duct length, and duct diameter improvements
- Richer reporting with recommendations, scenario comparisons, PDF export, and JSON export

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
- This demo uses **simulated data from Ventsim-inspired assumptions** (no real mine data).
- The ML model learns a correction factor `alpha` such that:

`C(t) = C0 * exp(-(Q/V) * alpha * t)`

where `Q` is airflow rate (m³/s) and `V` is the effective volume (m³).

## Folder structure
- `app.py` – Streamlit app
- `requirements.txt` – dependencies
- `assets/` – branding images (placeholder)
- `data/` – simulated dataset 

