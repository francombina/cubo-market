import json
import math
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

# Reemplazá por tu Gmail real
HEADERS = {"User-Agent": "Fran Combina francisco.combina@gmail.com"}

TICKERS_FILE = "tickers.json"
OUTPUT_FILE = "data/fundamentales.json"
HISTORY_YEARS = 10


def clean_num(val, decimals=None):
    """Convierte a float válido (evita NaN/Infinity que rompen JSON.parse)."""
    if val is None or isinstance(val, bool):
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return round(num, decimals) if decimals is not None else num


def safe_div(num, den, decimals=4):
    num_c = clean_num(num)
    den_c = clean_num(den)
    if num_c is None or den_c is None or den_c == 0:
        return None
    return round(num_c / den_c, decimals)


def get_ticker_cik_map() -> dict:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {
        item["ticker"].upper(): str(item["cik_str"]).zfill(10)
        for item in data.values()
    }


def get_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_sp500_monthly_returns():
    """Descarga 5 años de retornos mensuales del S&P 500 para respaldo de Beta."""
    try:
        hist = yf.Ticker("^GSPC").history(period="5y", interval="1mo")["Close"]
        ret = hist.pct_change().dropna()
        ret.index = ret.index.tz_localize(None).to_period("M")
        return ret
    except Exception as e:
        print(f"AVISO: No se pudo descargar ^GSPC para Beta de respaldo -> {e}")
        return None


def extract_annual_series(
    facts: dict,
    tag_specs: list,
    unit: str = "USD",
    is_duration: bool = True,
) -> dict:
    """
    Extrae toda la serie anual histórica (10-K / 10-K/A) combinando tags alternativos
    (por ejemplo, el cambio de tags por ASC 606 en 2018).
    Devuelve: { 'YYYY-MM-DD': (val, filed_str) }
    """
    all_facts = facts.get("facts", {})
    by_end = {}

    for priority, (taxonomy, tag) in enumerate(tag_specs):
        tag_data = all_facts.get(taxonomy, {}).get(tag, {})
        entries = tag_data.get("units", {}).get(unit, [])

        for entry in entries:
            if entry.get("form") not in ("10-K", "10-K/A"):
                continue

            end_str = entry.get("end")
            filed_str = entry.get("filed")
            val = clean_num(entry.get("val"))
            if not end_str or not filed_str or val is None:
                continue

            if is_duration:
                start_str = entry.get("start")
                if not start_str:
                    continue
                days = (
                    datetime.strptime(end_str, "%Y-%m-%d")
                    - datetime.strptime(start_str, "%Y-%m-%d")
                ).days
                # Filtra únicamente períodos fiscales anuales completos (~1 año)
                if days < 330 or days > 400:
                    continue

            rank = (filed_str, -priority)
            if end_str not in by_end or rank > by_end[end_str][0]:
                by_end[end_str] = (rank, val, filed_str)

    return {end_s: (val, filed_s) for end_s, (_, val, filed_s) in by_end.items()}


def lookup_near_date(series: dict, target_end: str, max_days: int = 15):
    """Busca el valor en una serie anual exactamente en target_end o dentro de +/- max_days."""
    if not series or not target_end:
        return None, None
    if target_end in series:
        return series[target_end]

    target_dt = datetime.strptime(target_end, "%Y-%m-%d")
    best_diff = max_days + 1
    best_item = (None, None)

    for end_s, item in series.items():
        diff = abs((datetime.strptime(end_s, "%Y-%m-%d") - target_dt).days)
        if diff < best_diff:
            best_diff = diff
            best_item = item

    return best_item


def build_anchor_dates(rev_series: dict, ni_series: dict) -> list:
    """Construye la lista ordenada (antiguo -> nuevo) de cierres anuales sin duplicados."""
    raw_dates = sorted(set(rev_series.keys()) | set(ni_series.keys()))
    anchors = []

    for d in raw_dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        if anchors:
            prev_dt = datetime.strptime(anchors[-1], "%Y-%m-%d")
            if (dt - prev_dt).days <= 20:
                # Si hay dos fechas pegadas, priorizar la que tenga Revenue
                if d in rev_series and anchors[-1] not in rev_series:
                    anchors[-1] = d
                continue
        anchors.append(d)

    return anchors


def get_market_data_and_history(ticker: str, sp500_ret):
    """
    Descarga info actual + precios diarios históricos (sin ajuste de dividendos,
    solo splits) + tabla de splits en una sola llamada a yfinance.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        # auto_adjust=False: Close viene ajustado por splits pero NO por dividendos
        # (fundamental para que el P/E histórico sea fiel al múltiplo real de cada año)
        hist_df = tk.history(period="max", interval="1d", auto_adjust=False)

        close_series = pd.Series(dtype=float)
        splits_list = []

        if hist_df is not None and not hist_df.empty:
            idx_naive = hist_df.index.tz_localize(None).normalize()
            if "Close" in hist_df.columns:
                close_series = pd.Series(hist_df["Close"].values, index=idx_naive).dropna()
                close_series = close_series[~close_series.index.duplicated(keep="last")].sort_index()

            if "Stock Splits" in hist_df.columns:
                sp = pd.Series(hist_df["Stock Splits"].values, index=idx_naive)
                sp = sp[sp > 0]
                splits_list = [(dt.date(), float(ratio)) for dt, ratio in sp.items()]

        beta = clean_num(info.get("beta"), 2)

        # Respaldo matemático de Beta (5 años mensual vs S&P 500)
        if beta is None and sp500_ret is not None and not close_series.empty:
            monthly_close = close_series.groupby(close_series.index.to_period("M")).last()
            stock_ret = monthly_close.pct_change().dropna().tail(60)
            aligned = stock_ret.to_frame("stock").join(
                sp500_ret.to_frame("mkt"), how="inner"
            )
            if len(aligned) >= 24:
                var_mkt = aligned["mkt"].var()
                if var_mkt and var_mkt != 0:
                    beta = clean_num(aligned["stock"].cov(aligned["mkt"]) / var_mkt, 2)

        return {
            "price": clean_num(info.get("currentPrice") or info.get("regularMarketPrice"), 2),
            "market_cap": clean_num(info.get("marketCap"), 0),
            "pe_ttm": clean_num(info.get("trailingPE"), 2),
            "forward_pe": clean_num(info.get("forwardPE"), 2),
            "yf_peg": clean_num(info.get("trailingPegRatio") or info.get("pegRatio"), 2),
            "beta": beta,
            "ev_to_ebitda": clean_num(info.get("enterpriseToEbitda"), 2),
            "yf_shares": clean_num(info.get("sharesOutstanding"), 0),
            "yf_eps": clean_num(info.get("trailingEps"), 2),
            "close_series": close_series,
            "splits": splits_list,
        }
    except Exception as e:
        print(f"AVISO yfinance en {ticker}: {e}")
        return {
            "price": None,
            "market_cap": None,
            "pe_ttm": None,
            "forward_pe": None,
            "yf_peg": None,
            "beta": None,
            "ev_to_ebitda": None,
            "yf_shares": None,
            "yf_eps": None,
            "close_series": pd.Series(dtype=float),
            "splits": [],
        }


def get_price_at_date(close_series: pd.Series, target_end: str):
    """Obtiene el precio de cierre en target_end (o el último día hábil previo hasta 10 días)."""
    if close_series is None or close_series.empty or not target_end:
        return None
    target_ts = pd.Timestamp(target_end)
    subset = close_series[close_series.index <= target_ts]
    if subset.empty:
        return None
    last_dt = subset.index[-1]
    if (target_ts - last_dt).days > 10:
        return None
    return clean_num(subset.iloc[-1], 2)


def adjust_eps_for_splits(eps_val, filed_str: str, splits: list):
    """
    Ajusta un EPS histórico por los stock splits ocurridos DESPUÉS de la fecha
    de presentación (filed) del 10-K que lo reportó.
    """
    if eps_val is None or not filed_str or not splits:
        return clean_num(eps_val, 2)
    filed_date = datetime.strptime(filed_str, "%Y-%m-%d").date()
    factor = 1.0
    for split_date, ratio in splits:
        if split_date > filed_date and ratio > 0:
            factor *= ratio
    return clean_num(eps_val / factor, 2)


def parse_company(ticker: str, cik: str, sp500_ret) -> dict:
    facts = get_company_facts(cik)
    mkt = get_market_data_and_history(ticker, sp500_ret)

    # 1. Extraer series anuales completas de la SEC
    rev_s = extract_annual_series(
        facts,
        [
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "Revenues"),
            ("us-gaap", "SalesRevenueNet"),
            ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
            ("us-gaap", "InterestAndDividendIncomeOperating"),
        ],
        is_duration=True,
    )
    op_s = extract_annual_series(facts, [("us-gaap", "OperatingIncomeLoss")], is_duration=True)
    ni_s = extract_annual_series(facts, [("us-gaap", "NetIncomeLoss")], is_duration=True)
    eps_s = extract_annual_series(
        facts,
        [
            ("us-gaap", "EarningsPerShareDiluted"),
            ("us-gaap", "EarningsPerShareBasic"),
        ],
        unit="USD/shares",
        is_duration=True,
    )
    assets_s = extract_annual_series(facts, [("us-gaap", "Assets")], is_duration=False)
    equity_s = extract_annual_series(
        facts,
        [
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ],
        is_duration=False,
    )
    debt_s = extract_annual_series(
        facts,
        [
            ("us-gaap", "LongTermDebtNoncurrent"),
            ("us-gaap", "LongTermDebt"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
        ],
        is_duration=False,
    )
    cash_s = extract_annual_series(
        facts,
        [
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
            ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        ],
        is_duration=False,
    )
    ocf_s = extract_annual_series(
        facts,
        [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
        is_duration=True,
    )
    capex_s = extract_annual_series(
        facts,
        [
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
            ("us-gaap", "PaymentsToAcquireProductiveAssets"),
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"),
            ("us-gaap", "PaymentsForCapitalImprovements"),
        ],
        is_duration=True,
    )
    shares_dur_s = extract_annual_series(
        facts,
        [
            ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
            ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
        ],
        unit="shares",
        is_duration=True,
    )
    shares_inst_s = extract_annual_series(
        facts,
        [
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
        ],
        unit="shares",
        is_duration=False,
    )

    # 2. Construir buffer de 13 años (para calcular CAGR de 3 años incluso en el año 1 del gráfico)
    anchors = build_anchor_dates(rev_s, ni_s)[-(HISTORY_YEARS + 3):]
    buffer = []

    for fy_end in anchors:
        revenue, _ = lookup_near_date(rev_s, fy_end)
        op_income, _ = lookup_near_date(op_s, fy_end)
        net_income, _ = lookup_near_date(ni_s, fy_end)
        raw_eps, eps_filed = lookup_near_date(eps_s, fy_end)
        assets, _ = lookup_near_date(assets_s, fy_end)
        equity, _ = lookup_near_date(equity_s, fy_end)
        debt, _ = lookup_near_date(debt_s, fy_end)
        cash, _ = lookup_near_date(cash_s, fy_end)
        ocf, _ = lookup_near_date(ocf_s, fy_end)
        capex, _ = lookup_near_date(capex_s, fy_end)

        sec_shares, _ = lookup_near_date(shares_dur_s, fy_end)
        if sec_shares is None:
            sec_shares, _ = lookup_near_date(shares_inst_s, fy_end, max_days=60)

        fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None
        adj_eps = adjust_eps_for_splits(raw_eps, eps_filed, mkt["splits"])
        price_at_end = get_price_at_date(mkt["close_series"], fy_end)

        hist_pe = None
        if price_at_end and adj_eps and adj_eps > 0:
            hist_pe = clean_num(price_at_end / adj_eps, 2)

        buffer.append(
            {
                "year": int(fy_end[:4]),
                "fy_end": fy_end,
                "price": price_at_end,
                "revenue": revenue,
                "operating_income": op_income,
                "net_income": net_income,
                "eps_diluted": adj_eps,
                "assets": assets,
                "equity": equity,
                "long_term_debt": debt,
                "cash": cash,
                "operating_cash_flow": ocf,
                "capex": capex,
                "free_cash_flow": fcf,
                "sec_shares": sec_shares,
                "pe": hist_pe,
                "operating_margin": safe_div(op_income, revenue),
                "net_margin": safe_div(net_income, revenue),
                "roe": safe_div(net_income, equity),
                "roa": safe_div(net_income, assets),
                "debt_to_equity": safe_div(debt, equity),
            }
        )

    # 3. Calcular EPS CAGR (3 años, con fallback a 2 años) y PEG histórico para cada punto
    for i, row in enumerate(buffer):
        cagr = None
        for lookback in (3, 2):
            if i >= lookback:
                eps_now = row["eps_diluted"]
                eps_prev = buffer[i - lookback]["eps_diluted"]
                if eps_now and eps_prev and eps_now > 0 and eps_prev > 0:
                    days = (
                        datetime.strptime(row["fy_end"], "%Y-%m-%d")
                        - datetime.strptime(buffer[i - lookback]["fy_end"], "%Y-%m-%d")
                    ).days
                    years = days / 365.25
                    if years >= lookback - 0.3:
                        cagr = clean_num((eps_now / eps_prev) ** (1 / years) - 1, 4)
                        break

        row["eps_cagr"] = cagr

        # PEG histórico (requiere crecimiento >= 1% para evitar valores infinitos en el gráfico)
        hist_peg = None
        if row["pe"] and row["pe"] > 0 and cagr is not None and cagr >= 0.01:
            hist_peg = clean_num(row["pe"] / (cagr * 100), 2)
        row["peg"] = hist_peg

    # 4. Armar historial final de 10 años (orden cronológico antiguo -> nuevo)
    history = [
        {
            "year": r["year"],
            "fy_end": r["fy_end"],
            "price": r["price"],
            "revenue": r["revenue"],
            "operating_income": r["operating_income"],
            "net_income": r["net_income"],
            "eps_diluted": r["eps_diluted"],
            "free_cash_flow": r["free_cash_flow"],
            "pe": r["pe"],
            "peg": r["peg"],
            "eps_cagr": r["eps_cagr"],
            "roe": r["roe"],
            "roa": r["roa"],
            "operating_margin": r["operating_margin"],
            "net_margin": r["net_margin"],
            "debt_to_equity": r["debt_to_equity"],
        }
        for r in buffer[-HISTORY_YEARS:]
    ]

    latest = buffer[-1] if buffer else {}

    # 5. Métricas actuales (compatibles 100% con tu Google Apps Script actual)
    latest_cagr = latest.get("eps_cagr")
    pe_ttm = mkt["pe_ttm"]
    peg_current = mkt["yf_peg"]
    if peg_current is None and pe_ttm and pe_ttm > 0 and latest_cagr and latest_cagr >= 0.01:
        peg_current = clean_num(pe_ttm / (latest_cagr * 100), 2)

    fcf_latest = latest.get("free_cash_flow")
    price_to_fcf = (
        safe_div(mkt["market_cap"], fcf_latest, 2)
        if (mkt["market_cap"] and fcf_latest and fcf_latest > 0)
        else None
    )

    eps_final = (
        latest.get("eps_diluted")
        if latest.get("eps_diluted") is not None
        else mkt["yf_eps"]
    )
    shares_final = mkt["yf_shares"] or latest.get("sec_shares")

    return {
        "ticker": ticker,
        "cik": cik,
        "name": facts.get("entityName", ""),
        "fy_end": latest.get("fy_end"),
        # Mercado y Valuación actual
        "price": mkt["price"],
        "market_cap": mkt["market_cap"],
        "pe_ttm": pe_ttm,
        "forward_pe": mkt["forward_pe"],
        "eps_cagr_2y": latest_cagr,
        "peg": peg_current,
        "beta": mkt["beta"],
        "price_to_fcf": price_to_fcf,
        "ev_to_ebitda": mkt["ev_to_ebitda"],
        # Último 10-K
        "revenue": latest.get("revenue"),
        "operating_income": latest.get("operating_income"),
        "net_income": latest.get("net_income"),
        "eps_diluted": clean_num(eps_final, 2),
        "assets": latest.get("assets"),
        "equity": latest.get("equity"),
        "long_term_debt": latest.get("long_term_debt"),
        "cash": latest.get("cash"),
        "operating_cash_flow": latest.get("operating_cash_flow"),
        "capex": latest.get("capex"),
        "free_cash_flow": fcf_latest,
        "shares_outstanding": shares_final,
        # Rentabilidad y Solvencia
        "operating_margin": latest.get("operating_margin"),
        "net_margin": latest.get("net_margin"),
        "roe": latest.get("roe"),
        "roa": latest.get("roa"),
        "debt_to_equity": latest.get("debt_to_equity"),
        # Serie histórica de 10 años para gráficos
        "history": history,
    }


def main():
    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
        tickers = json.load(f)

    cik_map = get_ticker_cik_map()
    sp500_ret = get_sp500_monthly_returns()

    results = []
    errors = []

    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper().replace(".", "-")
        cik = cik_map.get(ticker)
        if not cik:
            errors.append({"ticker": ticker, "error": "CIK no encontrado"})
            continue

        try:
            company_data = parse_company(ticker, cik, sp500_ret)
            results.append(company_data)
            print(
                f"OK: {ticker} | Años hist: {len(company_data['history'])} | "
                f"PE: {company_data['pe_ttm']} | PEG: {company_data['peg']} | "
                f"Beta: {company_data['beta']}"
            )
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
            print(f"ERROR: {ticker} -> {e}")

        time.sleep(0.25)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "count": len(results),
        "data": results,
        "errors": errors,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
