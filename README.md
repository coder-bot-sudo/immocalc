# immocalc

Minimaler Streamlit-Rechner, um den **Sweet Spot zwischen Kaufpreis und Miete** zu finden.

## Annahmen (transparent)
- Sollzins: **4% p.a.**
- Tilgung: **1,0% – 1,5% p.a.** (Band)
- Darlehen = Kaufpreis (Nebenkosten ignoriert in dieser Minimalversion)
- Objektwert konstant = Kaufpreis (keine Wertsteigerung)

## Start (lokal)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Dann im Browser öffnen (wird im Terminal angezeigt), typischerweise `http://localhost:8501`.

## Dateien
- `app.py`: Streamlit Dashboard (Kaufpreis/Miete Slider + Charts)
- `tool/immobilie_excel.py`: Excel-Generator (optional)
- `tool/immobilie_plots.py`: Matplotlib-Plots (optional)

## Hosting (einfach)
- Code auf GitHub pushen
- Dann z.B. **Streamlit Community Cloud** mit dem GitHub-Repo verbinden und als Entry-Point `app.py` wählen.
