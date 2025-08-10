# CryptoForecast — Final

ساختار پروژه:
- app.py : فایل اصلی استریم‌لیت
- requirements.txt : پکیج‌های لازم
- data/ : نمونهٔ داده‌ها و اخبار محلی
- news/news_utils.py : توابع گرفتن اخبار (NewsAPI یا fallback از CSV)
- config/settings.example.json : نمونهٔ تنظیمات (برای NewsAPI)

نحوه اجرا (لوکال):
1. نصب وابستگی‌ها:
   pip install -r requirements.txt

2. اجرا:
   streamlit run app.py

نکات:
- اگر می‌خواهی اخبار زنده داشته باشی، کلید NewsAPI را بگیر و در config/settings.json وارد کن:
  { "newsapi_key": "YOUR_KEY_HERE" }

- اگر Prophet نصب نشد یا خطا داد، می‌توانیم پیش‌بینی را به روش ساده‌تری موقتاً جایگزین کنیم.

- برای تست سریع: در سایدبار گزینه Upload CSV را استفاده کن و فایل data/sample.csv را بارگذاری کن.
