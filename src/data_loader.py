"""
News headline fetching module.

US mode priority order (auto):
  1. Alpha Vantage News Sentiment API  – real news, free tier 25 req/day
  2. yfinance ticker.news              – real recent headlines, no key needed
  3. Built-in US sample dataset        – offline fallback, always available

BIST mode priority order (auto):
  1. KAP RSS                           – public company disclosures (kap.org.tr)
  2. Bigpara ekonomi RSS               – Hurriyet financial news feed
  3. Investing.com Turkey RSS          – Turkish market news
  4. Built-in BIST sample dataset      – offline fallback, always available

Usage:
    from src.data_loader import load_news
    df = load_news("AAPL", days=180, market="US")
    df = load_news("THYAO", days=180, market="BIST")
"""

from __future__ import annotations

import os
import datetime
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
AV_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# On Streamlit Cloud, secrets are managed via st.secrets (not .env).
if not AV_API_KEY:
    try:
        import streamlit as st
        AV_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
    except Exception:
        pass

# ── HTTP headers for RSS requests ────────────────────────────────────────────
_RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

# ── BIST ticker → company keyword mapping ────────────────────────────────────
# Used to filter RSS feed items that are relevant to the selected ticker.
_BIST_KEYWORDS: dict[str, list[str]] = {
    "THYAO": ["thyao", "türk hava", "turk hava", "turkish airlines", "t.hava", "thy"],
    "ASELS": ["asels", "aselsan"],
    "GARAN": ["garan", "garanti"],
    "AKBNK": ["akbnk", "akbank"],
    "KCHOL": ["kchol", "koç holding", "koc holding", "koç"],
    "SISE":  ["sise", "şişecam", "sisecam", "şişe cam"],
    "EREGL": ["eregl", "ereğli", "eregli", "erdemir"],
    "TUPRS": ["tuprs", "tüpraş", "tupras"],
    "BIMAS": ["bimas", "bim birleşik", "bim mağaza", "bim "],
    "ISCTR": ["isctr", "iş bankası", "is bankasi", "işbank", "isbank"],
}

# ---------------------------------------------------------------------------
# US sample headline pool — 240 realistic headlines across major tickers
# ---------------------------------------------------------------------------
_HEADLINES = [
    # ── AAPL ──────────────────────────────────────────────────────────────
    ("Apple reports record quarterly revenue, beats analyst expectations by 8%", "AAPL", 1),
    ("Apple iPhone 15 sales surge in emerging markets, boosting Q1 outlook", "AAPL", 1),
    ("Apple announces $110B share buyback program, largest in company history", "AAPL", 1),
    ("Apple Vision Pro pre-orders exceed forecasts, analysts raise price targets", "AAPL", 1),
    ("Apple services revenue hits all-time high driven by App Store growth", "AAPL", 1),
    ("Apple Watch captures 30% of global smartwatch market, IDC report shows", "AAPL", 1),
    ("Apple faces antitrust scrutiny in EU over App Store payment rules", "AAPL", -1),
    ("Apple supply chain disruption in China threatens holiday shipments", "AAPL", -1),
    ("Apple revenue misses estimates as iPhone demand slows in greater China", "AAPL", -1),
    ("Apple under pressure as rival foldable phones gain traction in Asia", "AAPL", -1),
    ("Apple unveils iOS 18 with AI features at WWDC developer conference", "AAPL", 0),
    ("Apple holds annual shareholder meeting, executive compensation approved", "AAPL", 0),
    ("Apple expands manufacturing footprint in India amid China diversification", "AAPL", 0),
    ("Apple partners with IBM for enterprise AI solutions across Fortune 500", "AAPL", 0),
    ("Apple quarterly earnings call scheduled for next Tuesday", "AAPL", 0),
    # ── MSFT ──────────────────────────────────────────────────────────────
    ("Microsoft Azure cloud revenue grows 29% year-over-year in Q2", "MSFT", 1),
    ("Microsoft Copilot AI integration driving enterprise subscription growth", "MSFT", 1),
    ("Microsoft raises dividend by 10%, announces new $60B buyback plan", "MSFT", 1),
    ("Microsoft Teams surpasses 300 million daily active users globally", "MSFT", 1),
    ("Microsoft gaming division revenue up 61% following Activision acquisition", "MSFT", 1),
    ("Microsoft layoffs affect 1,900 employees in gaming and Azure divisions", "MSFT", -1),
    ("Microsoft faces regulatory pushback on AI monopoly concerns in Europe", "MSFT", -1),
    ("Microsoft cloud growth misses high Wall Street expectations for third quarter", "MSFT", -1),
    ("Microsoft announces new data centre in Malaysia expanding Asia presence", "MSFT", 0),
    ("Microsoft and OpenAI deepen partnership with extended $10B investment", "MSFT", 0),
    ("Microsoft releases quarterly security update patching 74 vulnerabilities", "MSFT", 0),
    ("Microsoft CEO Satya Nadella speaks at Davos about AI and productivity", "MSFT", 0),
    # ── TSLA ──────────────────────────────────────────────────────────────
    ("Tesla delivers record 485,000 vehicles in Q3, manufacturing efficiency improves", "TSLA", 1),
    ("Tesla Cybertruck production ramp ahead of schedule, pre-orders strong", "TSLA", 1),
    ("Tesla Full Self-Driving subscription revenue growing faster than expected", "TSLA", 1),
    ("Tesla opens massive Gigafactory in Mexico, doubling global capacity", "TSLA", 1),
    ("Tesla energy storage business hits record revenue, margins expanding", "TSLA", 1),
    ("Tesla recalls 2 million vehicles over autopilot safety concerns in US", "TSLA", -1),
    ("Tesla misses delivery estimates as aggressive price cuts pressure margins", "TSLA", -1),
    ("Elon Musk distraction at Twitter weighs on Tesla investor confidence", "TSLA", -1),
    ("Tesla faces stiff competition from BYD in China, market share slipping", "TSLA", -1),
    ("Tesla gross margin falls to 17.4%, below analyst expectations of 18.5%", "TSLA", -1),
    ("Tesla opens new service centre network in Southeast Asian markets", "TSLA", 0),
    ("Tesla annual shareholder meeting approves CEO pay package after recount", "TSLA", 0),
    ("Tesla updates software over-the-air improving range estimates by 3%", "TSLA", 0),
    # ── GOOGL ─────────────────────────────────────────────────────────────
    ("Alphabet beats earnings estimates, ad revenue rebounds strongly in Q4", "GOOGL", 1),
    ("Google Cloud posts first-ever quarterly profit, shares jump 6%", "GOOGL", 1),
    ("Google Gemini AI model outperforms GPT-4 on multiple benchmarks", "GOOGL", 1),
    ("YouTube ad revenue up 12% year-over-year, monetisation improving", "GOOGL", 1),
    ("Google hit with $5B EU antitrust fine over Android search practices", "GOOGL", -1),
    ("Google misses ad revenue estimates as TikTok competition intensifies", "GOOGL", -1),
    ("Google faces DOJ lawsuit over search monopoly, landmark case begins", "GOOGL", -1),
    ("Google updates search algorithm, publishers report traffic declines", "GOOGL", -1),
    ("Google announces Pixel 9 lineup at Made by Google hardware event", "GOOGL", 0),
    ("Google releases Bard enterprise tier for businesses at competitive pricing", "GOOGL", 0),
    # ── AMZN ──────────────────────────────────────────────────────────────
    ("Amazon AWS revenue accelerates 17%, margin expansion impresses investors", "AMZN", 1),
    ("Amazon Prime membership crosses 200 million globally, ARPU rising", "AMZN", 1),
    ("Amazon advertising segment grows 26% becoming third-largest digital ad platform", "AMZN", 1),
    ("Amazon same-day delivery now covers 80% of US metropolitan areas", "AMZN", 1),
    ("Amazon warehouse workers strike disrupts peak-season fulfilment in UK", "AMZN", -1),
    ("Amazon operating income falls as heavy investment cycle accelerates", "AMZN", -1),
    ("Amazon faces FTC antitrust lawsuit over Prime subscription practices", "AMZN", -1),
    ("Amazon launches new fulfilment centre in Texas creating 3,000 jobs", "AMZN", 0),
    ("Amazon and Stellantis expand Alexa in-car integration partnership", "AMZN", 0),
    # ── NVDA ──────────────────────────────────────────────────────────────
    ("NVIDIA reports blowout earnings, revenue triples on AI chip demand", "NVDA", 1),
    ("NVIDIA H100 GPU backlog extends to 12 months as hyperscalers expand", "NVDA", 1),
    ("NVIDIA announces Blackwell B200 GPU delivering 30x AI inference speedup", "NVDA", 1),
    ("NVIDIA data centre segment revenue hits $18.4B, beating all forecasts", "NVDA", 1),
    ("NVIDIA stock becomes third company to reach $2 trillion market cap", "NVDA", 1),
    ("NVIDIA export restrictions to China could cost $10B in annual revenue", "NVDA", -1),
    ("NVIDIA faces class-action lawsuit over alleged crypto revenue misleading", "NVDA", -1),
    ("NVIDIA supply chain constraints limit ability to meet AI server demand", "NVDA", -1),
    ("NVIDIA announces next-generation Grace Blackwell superchip platform", "NVDA", 0),
    ("NVIDIA and Oracle expand partnership on sovereign AI cloud infrastructure", "NVDA", 0),
    # ── SPY / Macro ───────────────────────────────────────────────────────
    ("S&P 500 hits new all-time high on strong jobs data and cooling inflation", "SPY", 1),
    ("Fed signals rate cuts on horizon as inflation nears 2% target", "SPY", 1),
    ("US economy adds 303,000 jobs in March, unemployment falls to 3.8%", "SPY", 1),
    ("Retail sales beat forecasts for third consecutive month, consumer resilient", "SPY", 1),
    ("Consumer confidence index rises to highest level since December 2021", "SPY", 1),
    ("Goldman Sachs upgrades tech sector to overweight on AI growth thesis", "SPY", 1),
    ("US GDP growth beats expectations at 3.1% annualised in Q3 2024", "SPY", 1),
    ("S&P 500 drops 2.3% as inflation data comes in hotter than expected", "SPY", -1),
    ("Federal Reserve holds rates higher for longer, markets sell off sharply", "SPY", -1),
    ("US GDP growth slows to 1.6% in Q1, below consensus estimate of 2.4%", "SPY", -1),
    ("Tech stocks tumble as 10-year Treasury yield hits 5% for first time since 2007", "SPY", -1),
    ("Inflation stays elevated at 3.5%, dashing near-term rate-cut hopes", "SPY", -1),
    ("China economic slowdown weighs on global growth and commodity prices", "SPY", -1),
    ("Commercial real estate crisis deepens, banks raise loan-loss provisions", "SPY", -1),
    ("Federal Reserve keeps interest rates unchanged at 5.25–5.50%", "SPY", 0),
    ("US CPI inflation in line with economist consensus at 3.4%", "SPY", 0),
    ("Federal Reserve releases FOMC meeting minutes, no new policy signals", "SPY", 0),
    ("SEC announces review of AI-related disclosures in corporate filings", "SPY", 0),
    ("US Treasury auctions 10-year notes at 4.62% yield, demand in line", "SPY", 0),
    ("IMF revises global growth forecast to 3.2%, slight upward revision", "SPY", 0),
    ("S&P 500 ends flat as investors await Fed Chair Powell speech Friday", "SPY", 0),
    ("OPEC+ maintains current oil production levels at monthly Vienna meeting", "SPY", 0),
]

# ---------------------------------------------------------------------------
# BIST sample headline pool — Turkish-language realistic headlines
# Each entry: (headline, primary_ticker, rough_sentiment  1=pos 0=neu -1=neg)
# ---------------------------------------------------------------------------
_BIST_HEADLINES = [
    # ── THYAO ─────────────────────────────────────────────────────────────
    ("Türk Hava Yolları üçüncü çeyrekte rekor yolcu sayısına ulaştı", "THYAO", 1),
    ("THY net kârı yıllık yüzde 38 artışla beklentileri aştı", "THYAO", 1),
    ("Türk Hava Yolları Avrupa'ya 12 yeni destinasyon ekliyor, talepler güçlü", "THYAO", 1),
    ("THY filo büyüme planı kapsamında 40 yeni uçak siparişi verdi", "THYAO", 1),
    ("THY yakıt maliyetlerindeki artış nedeniyle marjlar baskı altına girdi", "THYAO", -1),
    ("Türk Hava Yolları kargo gelirlerinde belirgin düşüş yaşandığını açıkladı", "THYAO", -1),
    ("THY döviz kuru volatilitesi nedeniyle kârlılık riski gündemde", "THYAO", -1),
    ("Türk Hava Yolları yeni Orta Doğu seferlerine başladığını duyurdu", "THYAO", 0),
    ("THY 2024 yatırım bütçesini ve filo planlarını açıkladı", "THYAO", 0),
    # ── ASELS ─────────────────────────────────────────────────────────────
    ("Aselsan savunma ihracatında rekor gelire ulaşarak büyüme sürdürdü", "ASELS", 1),
    ("Aselsan yurt dışı savunma sözleşmelerinde büyük artış kaydetti", "ASELS", 1),
    ("Aselsan yeni nesil radar sistemi için önemli ihracat sözleşmesi imzaladı", "ASELS", 1),
    ("Aselsan güçlü kamu sözleşmeleriyle kârlılığını artırdı", "ASELS", 1),
    ("Aselsan bazı projelerdeki gecikme nakit akışını olumsuz etkiliyor", "ASELS", -1),
    ("Aselsan hammadde maliyetlerindeki yükseliş marjları daralttı", "ASELS", -1),
    ("Aselsan uluslararası iş birliği kapsamını genişletti", "ASELS", 0),
    ("Aselsan Türk Silahlı Kuvvetleri ile yeni tedarik protokolü imzaladı", "ASELS", 0),
    # ── GARAN ─────────────────────────────────────────────────────────────
    ("Garanti BBVA net kârı beklentilerin üzerinde açıklandı", "GARAN", 1),
    ("Garanti Bankası dijital müşteri tabanı 14 milyonu aştı", "GARAN", 1),
    ("Garanti BBVA temettü artışı açıkladı, yatırımcılar memnuniyetini dile getirdi", "GARAN", 1),
    ("Garanti Bankası kredi büyümesi sektör ortalamasının belirgin üzerinde seyretti", "GARAN", 1),
    ("Garanti BBVA takipteki kredi oranında sınırlı bir yükseliş gözlemlendi", "GARAN", -1),
    ("Garanti Bankası faiz marjlarının daralmaya devam ettiğini açıkladı", "GARAN", -1),
    ("Garanti BBVA sürdürülebilir finans alanında yeni hedeflerini paylaştı", "GARAN", 0),
    ("Garanti Bankası olağan genel kurulu tamamlandı", "GARAN", 0),
    # ── AKBNK ─────────────────────────────────────────────────────────────
    ("Akbank güçlü çeyrek kârıyla özkaynak kârlılığını yüksek tuttu", "AKBNK", 1),
    ("Akbank dijital bankacılık altyapısına yönelik yatırımlarını artırıyor", "AKBNK", 1),
    ("Akbank yüksek faiz ortamında güçlü net faiz marjını korudu", "AKBNK", 1),
    ("Akbank kredi kartı işlem hacminde rekor kırdı", "AKBNK", 1),
    ("Akbank operasyonel maliyetler baskı altında seyrediyor", "AKBNK", -1),
    ("Akbank döviz kuru riski yönetiminde zorluklarla karşılaşıyor", "AKBNK", -1),
    ("Akbank yeni dijital şube ağı genişleme stratejisini açıkladı", "AKBNK", 0),
    ("Akbank kurumsal bankacılık segmentinde büyümeyi sürdürüyor", "AKBNK", 0),
    # ── KCHOL ─────────────────────────────────────────────────────────────
    ("Koç Holding konsolide kârı analist tahminlerini aştı", "KCHOL", 1),
    ("Koç Holding enerji ve otomotiv segmentlerinde güçlü büyüme kaydetti", "KCHOL", 1),
    ("Koç Holding yenilenebilir enerji yatırımlarını hızlandırıyor", "KCHOL", 1),
    ("Koç Holding bağlı ortaklıklardan gelen temettü gelirleri arttı", "KCHOL", 1),
    ("Koç Holding bazı segmentlerde marj daralması yaşandığını açıkladı", "KCHOL", -1),
    ("Koç Holding yurt dışı makroekonomik risklerden etkilenebileceğini belirtti", "KCHOL", -1),
    ("Koç Holding yıllık stratejik plan toplantısını tamamladı", "KCHOL", 0),
    ("Koç Holding yönetim kurulunda görev dağılımı güncellendi", "KCHOL", 0),
    # ── SISE ──────────────────────────────────────────────────────────────
    ("Şişecam cam ve kimyasallar segmentinde güçlü büyüme kaydetti", "SISE", 1),
    ("Şişecam ihracat gelirlerinde yüzde 22 artış elde ettiğini açıkladı", "SISE", 1),
    ("Şişecam Avrupa'da yeni üretim tesisi açarak kapasiteyi genişletti", "SISE", 1),
    ("Şişecam enerji maliyetlerindeki yükseliş kârlılığı olumsuz etkiliyor", "SISE", -1),
    ("Şişecam Avrupa cam talebinde yavaşlama gözlemleniyor", "SISE", -1),
    ("Şişecam soda külü kapasite artırım yatırımlarına devam ettiğini bildirdi", "SISE", 0),
    ("Şişecam sürdürülebilirlik ve çevre hedefleri güncellendi", "SISE", 0),
    # ── EREGL ─────────────────────────────────────────────────────────────
    ("Ereğli Demir Çelik güçlü yurt içi talep sayesinde satışlarını artırdı", "EREGL", 1),
    ("Erdemir ihracat fiyatlarındaki iyileşmeyle kârlılığını güçlendirdi", "EREGL", 1),
    ("Ereğli çelik üretiminde rekor kapasiteye ulaştı", "EREGL", 1),
    ("Erdemir ithal çelik rekabeti fiyatlar üzerinde baskı oluşturuyor", "EREGL", -1),
    ("Ereğli enerji ve hammadde maliyetleri marjı sıkıştırmaya devam ediyor", "EREGL", -1),
    ("Erdemir çevre uyum yatırımları kapsamında önemli harcama yapıldığını açıkladı", "EREGL", -1),
    ("Ereğli Demir Çelik çelik teslimat programını revize etti", "EREGL", 0),
    ("Erdemir katma değerli çelik üretim kapasitesini artırıyor", "EREGL", 0),
    # ── TUPRS ─────────────────────────────────────────────────────────────
    ("Tüpraş güçlü rafineri marjlarıyla rekor kâr elde ettiğini açıkladı", "TUPRS", 1),
    ("Tüpraş ham petrol işleme kapasitesini artırarak verimliliği yükseltti", "TUPRS", 1),
    ("Tüpraş yenilenebilir yakıt alanındaki yatırımlarını duyurdu", "TUPRS", 1),
    ("Tüpraş ham petrol fiyatlarındaki volatilite marjı olumsuz etkiliyor", "TUPRS", -1),
    ("Tüpraş planlı bakım duruşu üretimde geçici düşüşe yol açtı", "TUPRS", -1),
    ("Tüpraş yıllık kapasite kullanım oranı ve operasyonel verileri açıklandı", "TUPRS", 0),
    ("Tüpraş enerji verimliliği yatırım programını hayata geçirdi", "TUPRS", 0),
    # ── BIMAS ─────────────────────────────────────────────────────────────
    ("BİM Birleşik Mağazalar yurt içi satışlarında güçlü büyüme kaydetti", "BIMAS", 1),
    ("BİM Mağazaları yurt dışı genişlemesini hızlandırdığını açıkladı", "BIMAS", 1),
    ("BİM net kârı beklentilerin üzerinde geldi, pazar payı artmaya devam ediyor", "BIMAS", 1),
    ("BİM enflasyonist ortamda lojistik ve pazarlama giderleri baskı oluşturuyor", "BIMAS", -1),
    ("BİM tedarik zinciri maliyetleri yüksek seyretmeye devam ediyor", "BIMAS", -1),
    ("BİM Mağazaları yeni mağaza açılış hedefini yıllık plana ekledi", "BIMAS", 0),
    ("BİM Mağazaları özel markalı ürün yelpazesini genişletiyor", "BIMAS", 0),
    # ── ISCTR ─────────────────────────────────────────────────────────────
    ("İş Bankası üçüncü çeyrekte net kârını yüzde 31 artırdı", "ISCTR", 1),
    ("İş Bankası güçlü mevduat büyümesiyle fonlama maliyetini aşağı çekti", "ISCTR", 1),
    ("İşbank dijital platformda işlem hacmi rekor seviyeye ulaştı", "ISCTR", 1),
    ("İş Bankası yüksek enflasyon ortamında marj baskısıyla karşı karşıya kaldı", "ISCTR", -1),
    ("İş Bankası takipteki kredi karşılık oranını artırdı", "ISCTR", -1),
    ("İş Bankası yıllık olağan genel kurulu tamamlandı, temettü onaylandı", "ISCTR", 0),
    ("İş Bankası kurumsal sosyal sorumluluk ve sürdürülebilirlik raporu yayımlandı", "ISCTR", 0),
    # ── BIST Makro ────────────────────────────────────────────────────────
    ("TCMB faiz kararı piyasa beklentileriyle örtüştü, BIST-100 yatay kapandı", "BIST", 0),
    ("Türkiye enflasyonu beklentilerin altında geldi, piyasalarda iyimserlik güçlendi", "BIST", 1),
    ("BIST-100 endeksi yabancı yatırımcı alımlarıyla yeni rekor tazeledi", "BIST", 1),
    ("Türkiye cari açığı daralıyor, ekonomistler olumlu değerlendiriyor", "BIST", 1),
    ("Jeopolitik riskler BIST üzerinde kısa vadeli satış baskısı oluşturdu", "BIST", -1),
    ("Türk lirası dolar karşısında değer kaybı yaşadı", "BIST", -1),
    ("Türkiye büyüme verileri beklentilerin altında kaldı, piyasalar düştü", "BIST", -1),
    ("SPK yeni düzenleme taslağını kamuoyuyla paylaştı, görüşe açıldı", "BIST", 0),
    ("BIST yabancı yatırımcı kayıt sayısında yeni rekor açıklandı", "BIST", 1),
    ("Türkiye hazine ihalesi güçlü talep gördü, faiz beklentilerin altında kaldı", "BIST", 1),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_pubdate(raw: str) -> pd.Timestamp:
    """Convert an RSS pubDate / Atom updated string to a tz-naive Timestamp."""
    try:
        ts = pd.to_datetime(raw)
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        return ts
    except Exception:
        return pd.Timestamp.now()


def _parse_rss_feed(
    url: str,
    ticker: str,
    source_label: str,
    keywords: list[str],
) -> pd.DataFrame:
    """
    Generic RSS/Atom feed fetcher and keyword filter.

    Fetches `url`, keeps items whose title contains at least one keyword from
    `keywords` (case-insensitive), and returns a normalised DataFrame with
    columns: date, headline, ticker, source.

    Passing an empty `keywords` list returns all items unfiltered.
    """
    resp = requests.get(url, timeout=15, headers=_RSS_HEADERS)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)

    # Support both RSS 2.0 (<channel><item>) and Atom (<feed><entry>) formats
    ns_atom = "http://www.w3.org/2005/Atom"
    channel = root.find("channel") or root
    items = channel.findall("item") or root.findall(f".//{{{ns_atom}}}entry")

    rows = []
    for item in items:
        title = (
            item.findtext("title")
            or item.findtext(f"{{{ns_atom}}}title")
            or ""
        ).strip()
        pub = (
            item.findtext("pubDate")
            or item.findtext(f"{{{ns_atom}}}updated")
            or item.findtext(f"{{{ns_atom}}}published")
            or ""
        ).strip()

        if not title:
            continue
        if keywords and not any(kw in title.lower() for kw in keywords):
            continue

        rows.append({
            "date":     _parse_pubdate(pub),
            "headline": title,
            "ticker":   ticker,
            "source":   source_label,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ── US news adapters ──────────────────────────────────────────────────────────

def _build_sample_dataset(ticker: str, days: int = 180) -> pd.DataFrame:
    """Generate a demo dataset from the US sample headline pool."""
    np.random.seed(42)
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    dates = pd.date_range(start, end, freq="B")

    pool = [h for h in _HEADLINES if h[1] in (ticker, "SPY")] or _HEADLINES

    rows = []
    for date in dates:
        n = np.random.randint(1, 5)
        idxs = np.random.choice(len(pool), size=n, replace=True)
        for i in idxs:
            headline, _, _ = pool[i]
            rows.append({
                "date":     pd.Timestamp(date),
                "headline": headline,
                "ticker":   ticker,
                "source":   "Sample Dataset",
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "headline"])
    return df.sort_values("date").reset_index(drop=True)


def _fetch_alphavantage(ticker: str, limit: int = 200) -> pd.DataFrame:
    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={ticker}"
        f"&limit={limit}&sort=LATEST&apikey={AV_API_KEY}"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if "feed" not in data:
        raise ValueError(f"Alpha Vantage response missing 'feed': {data}")

    rows = []
    for item in data["feed"]:
        try:
            date = pd.to_datetime(item["time_published"], format="%Y%m%dT%H%M%S")
        except Exception:
            continue
        rows.append({
            "date":     date,
            "headline": item.get("title", ""),
            "ticker":   ticker,
            "source":   item.get("source", "Alpha Vantage"),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _fetch_yfinance(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    rows = []
    for item in (t.news or []):
        ts = item.get("providerPublishTime", 0)
        title = item.get("title", "")
        if not title:
            continue
        rows.append({
            "date":     pd.Timestamp(ts, unit="s"),
            "headline": title,
            "ticker":   ticker,
            "source":   item.get("publisher", "yfinance"),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ── BIST news adapters ────────────────────────────────────────────────────────

def _fetch_kap_rss(ticker: str) -> pd.DataFrame:
    """
    KAP (Kamuyu Aydınlatma Platformu) — Turkey's public company disclosure platform.
    RSS feed contains regulatory filings and company announcements.
    Filters items by ticker-specific company keywords.
    """
    keywords = _BIST_KEYWORDS.get(ticker.upper(), [ticker.lower()])
    return _parse_rss_feed(
        "https://www.kap.org.tr/tr/rss",
        ticker,
        "KAP",
        keywords,
    )


def _fetch_bigpara(ticker: str) -> pd.DataFrame:
    """
    Bigpara (Hurriyet) — Turkish financial and economic news RSS feed.
    Filters items by ticker-specific company keywords.
    """
    keywords = _BIST_KEYWORDS.get(ticker.upper(), [ticker.lower()])
    return _parse_rss_feed(
        "https://bigpara.hurriyet.com.tr/rss/",
        ticker,
        "Bigpara",
        keywords,
    )


def _fetch_investing_tr(ticker: str) -> pd.DataFrame:
    """
    Investing.com Turkey — Turkish market and company news RSS feed.
    Uses the Turkey-specific news category feed.
    Filters items by ticker-specific company keywords.
    """
    keywords = _BIST_KEYWORDS.get(ticker.upper(), [ticker.lower()])
    return _parse_rss_feed(
        "https://tr.investing.com/rss/news_285.rss",
        ticker,
        "Investing.com TR",
        keywords,
    )


def _build_bist_sample_dataset(ticker: str, days: int = 180) -> pd.DataFrame:
    """
    Generate a demo dataset from the BIST Turkish-language sample headline pool.

    Filtering rules (strictest first):
      1. Use only headlines whose primary_ticker == ticker (exact company match).
      2. If none found, fall back to all company-specific headlines
         (h[1] != "BIST") — never injects macro/market-wide headlines.

    Macro/market-wide headlines (primary_ticker == "BIST") are deliberately
    excluded from ticker pages. Use get_bist_market_context() for those.
    """
    np.random.seed(42)
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    dates = pd.date_range(start, end, freq="B")

    # Rule 1 — exact ticker match only
    pool = [h for h in _BIST_HEADLINES if h[1] == ticker]

    # Rule 2 — fallback: all company headlines, macro excluded
    if not pool:
        pool = [h for h in _BIST_HEADLINES if h[1] != "BIST"]

    # Ultimate safety net (should never be reached with current data)
    if not pool:
        pool = _BIST_HEADLINES

    rows = []
    for date in dates:
        n = np.random.randint(1, 4)
        idxs = np.random.choice(len(pool), size=n, replace=True)
        for i in idxs:
            headline, _, _ = pool[i]
            rows.append({
                "date":     pd.Timestamp(date),
                "headline": headline,
                "ticker":   ticker,
                "source":   "BIST Sample Dataset",
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "headline"])
    return df.sort_values("date").reset_index(drop=True)


def _load_bist_news(ticker: str, days: int = 180, source: str = "auto") -> pd.DataFrame:
    """
    BIST news pipeline: KAP → Bigpara → Investing.com TR → BIST sample fallback.
    Each live source is tried in order; the first non-empty result is returned.
    On any network failure the next source is attempted transparently.
    """
    if source in ("kap", "auto"):
        try:
            df = _fetch_kap_rss(ticker)
            if not df.empty:
                print(f"  BIST news source: KAP ({len(df)} headlines)")
                return df
        except Exception as e:
            print(f"  KAP RSS failed: {e}")

    if source in ("bigpara", "auto"):
        try:
            df = _fetch_bigpara(ticker)
            if not df.empty:
                print(f"  BIST news source: Bigpara ({len(df)} headlines)")
                return df
        except Exception as e:
            print(f"  Bigpara failed: {e}")

    if source in ("investing_tr", "auto"):
        try:
            df = _fetch_investing_tr(ticker)
            if not df.empty:
                print(f"  BIST news source: Investing.com TR ({len(df)} headlines)")
                return df
        except Exception as e:
            print(f"  Investing.com TR failed: {e}")

    print("  BIST news source: built-in BIST sample dataset")
    return _build_bist_sample_dataset(ticker, days=days)


# ── public API ────────────────────────────────────────────────────────────────

def get_bist_market_context(days: int = 180) -> pd.DataFrame:
    """
    Return BIST-wide macro / market-context headlines from the sample pool.

    These are intentionally separated from company-specific ticker data.
    Use this to display a market overview panel alongside individual ticker
    analysis — never mix into a single company's scored headlines.

    Returns
    -------
    DataFrame with columns: date, headline, ticker ('BIST'), source
    """
    np.random.seed(0)
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    dates = pd.date_range(start, end, freq="B")

    pool = [h for h in _BIST_HEADLINES if h[1] == "BIST"]
    if not pool:
        return pd.DataFrame(columns=["date", "headline", "ticker", "source"])

    rows = []
    for date in dates:
        n = np.random.randint(0, 2)          # sparser than company feeds
        if n == 0:
            continue
        idxs = np.random.choice(len(pool), size=n, replace=True)
        for i in idxs:
            headline, _, _ = pool[i]
            rows.append({
                "date":     pd.Timestamp(date),
                "headline": headline,
                "ticker":   "BIST",
                "source":   "BIST Market Context",
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "headline"])
    return df.sort_values("date").reset_index(drop=True)


def load_news(
    ticker: str,
    days: int = 180,
    source: str = "auto",
    market: str = "US",
) -> pd.DataFrame:
    """
    Fetch news headlines for a ticker.

    Parameters
    ----------
    ticker : str    e.g. 'AAPL' or 'THYAO'
    days   : int    lookback window used by sample/yfinance adapters
    source : str    US:   'auto' | 'alphavantage' | 'yfinance' | 'sample'
                    BIST: 'auto' | 'kap' | 'bigpara' | 'investing_tr' | 'sample'
    market : str    'US' | 'BIST'  — controls which adapter pipeline is used

    Returns
    -------
    DataFrame with columns: date, headline, ticker, source
    """
    if market == "BIST":
        return _load_bist_news(ticker, days=days, source=source)

    # ── US pipeline ──────────────────────────────────────────────────────
    if source in ("alphavantage", "auto") and AV_API_KEY:
        try:
            df = _fetch_alphavantage(ticker)
            if not df.empty:
                print(f"  News source: Alpha Vantage ({len(df)} headlines)")
                return df
        except Exception as e:
            print(f"  Alpha Vantage failed: {e}")

    if source in ("yfinance", "auto"):
        try:
            df = _fetch_yfinance(ticker)
            if not df.empty:
                print(f"  News source: yfinance ({len(df)} headlines)")
                return df
        except Exception as e:
            print(f"  yfinance news failed: {e}")

    print("  News source: built-in US sample dataset")
    return _build_sample_dataset(ticker, days=days)
