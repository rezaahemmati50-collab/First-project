# CryptoForecast — Final Integrated

## Quick start (local)
1. Create virtual env:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

2. Install:
   pip install -r requirements.txt

3. Run:
   streamlit run app.py

## Notes
- Upload CSV must have columns: ds (date), y (price).
- If Prophet install fails, disable "Enable Prophet forecasting" in sidebar.
- For live news, place NewsAPI key in config/settings.json
