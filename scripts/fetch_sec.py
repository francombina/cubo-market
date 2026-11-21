import json
import os
import time
from datetime import datetime, timezone
import requests

# Reemplazá con el nombre de tu proyecto y tu Gmail real
HEADERS = {"User-Agent": "Fran Combina francisco.combina@gmail.com"}

TICKERS_FILE = "tickers.json"
OUTPUT_FILE = "data/fundamentales.json"


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


def extract_latest_fy(facts: dict, tag_specs: list, unit: str = "USD", is_duration: bool = True):
    """
    Busca el dato anual (10-K) más reciente entre varios tags alternativos.
    tag_specs: lista de tuplas (taxonomia, tag) ordenadas por prioridad.
    """
    all_facts = facts.get("facts", {})
    candidates = []

    for priority, (taxonomy, tag) in enumerate(tag_specs):
        tag_data = all_facts.get(taxonomy, {}).get(tag, {})
        entries = tag_data.get("units", {}).get(unit, [])

        for entry in entries:
            if entry.get("form") not in ("10-K", "10-K/A"):
                continue

            end_str = entry.get("end")
            filed_str = entry.get("filed")
            val = entry.get("val")
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
                if days < 330:
                    continue

            # Prioriza: fecha de cierre más nueva, luego presentación más nueva, luego prioridad del tag
            candidates.append((end_str, filed_str, -priority, val))

    if not candidates:
        return None, None

    best = max(candidates)
    return best[3], best[0]


def safe_div(num, den):
    if num is None or den is None or den == 0:
        return None
    return round(num / den, 4)


def parse_company(ticker: str, cik: str) -> dict:
    facts = get_company_facts(cik)

    revenue, fy_end = extract_latest_fy(
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

    op_income, _ = extract_latest_fy(
        facts,
        [("us-gaap", "OperatingIncomeLoss")],
        is_duration=True,
    )

    net_income, ni_end = extract_latest_fy(
        facts,
        [("us-gaap", "NetIncomeLoss")],
        is_duration=True,
    )

    eps_diluted, _ = extract_latest_fy(
        facts,
        [
            ("us-gaap", "EarningsPerShareDiluted"),
            ("us-gaap", "EarningsPerShareBasic"),
        ],
        unit="USD/shares",
        is_duration=True,
    )

    assets, _ = extract_latest_fy(
        facts,
        [("us-gaap", "Assets")],
        is_duration=False,
    )

    equity, _ = extract_latest_fy(
        facts,
        [
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ],
        is_duration=False,
    )

    long_term_debt, _ = extract_latest_fy(
        facts,
        [
            ("us-gaap", "LongTermDebtNoncurrent"),
            ("us-gaap", "LongTermDebt"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
        ],
        is_duration=False,
    )

    cash, _ = extract_latest_fy(
        facts,
        [
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
            ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        ],
        is_duration=False,
    )

    ocf, _ = extract_latest_fy(
        facts,
        [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
        is_duration=True,
    )

    capex, _ = extract_latest_fy(
        facts,
        [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment")],
        is_duration=True,
    )

    shares, _ = extract_latest_fy(
        facts,
        [
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
        ],
        unit="shares",
        is_duration=False,
    )

    fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None

    return {
        "ticker": ticker,
        "cik": cik,
        "name": facts.get("entityName", ""),
        "fy_end": fy_end or ni_end,
        "revenue": revenue,
        "operating_income": op_income,
        "net_income": net_income,
        "eps_diluted": eps_diluted,
        "assets": assets,
        "equity": equity,
        "long_term_debt": long_term_debt,
        "cash": cash,
        "operating_cash_flow": ocf,
        "capex": capex,
        "free_cash_flow": fcf,
        "shares_outstanding": shares,
        # Ratios en decimal (ej. 0.253 = 25.3% al formatear en Sheets)
        "operating_margin": safe_div(op_income, revenue),
        "net_margin": safe_div(net_income, revenue),
        "roe": safe_div(net_income, equity),
        "roa": safe_div(net_income, assets),
        "debt_to_equity": safe_div(long_term_debt, equity),
    }


def main():
    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
        tickers = json.load(f)

    cik_map = get_ticker_cik_map()
    results = []
    errors = []

    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper().replace(".", "-")
        cik = cik_map.get(ticker)
        if not cik:
            errors.append({"ticker": ticker, "error": "CIK no encontrado"})
            continue

        try:
            company_data = parse_company(ticker, cik)
            results.append(company_data)
            print(f"OK: {ticker} (cierre {company_data['fy_end']})")
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
            print(f"ERROR: {ticker} -> {e}")

        # Respetar límite de la SEC (máx 10 req/seg)
        time.sleep(0.2)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "count": len(results),
        "data": results,
        "errors": errors,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
    

import yfinance as yf


def get_sp500_monthly_returns():
    """Descarga una sola vez los retornos mensuales de 5 años del S&P 500."""
    hist = yf.Ticker("^GSPC").history(period="5y", interval="1mo")["Close"]
    return hist.pct_change().dropna()


def extract_eps_cagr_3y(facts: dict):
    """Calcula el CAGR de 3 años del EPS diluido usando los 10-K de la SEC."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    by_end = {}

    for tag in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
        entries = us_gaap.get(tag, {}).get("units", {}).get("USD/shares", [])
        for entry in entries:
            if entry.get("form") not in ("10-K", "10-K/A"):
                continue
            start_s, end_s, filed_s, val = (
                entry.get("start"), entry.get("end"), entry.get("filed"), entry.get("val")
            )
            if not (start_s and end_s and filed_s) or val is None:
                continue
            days = (datetime.strptime(end_s, "%Y-%m-%d") - datetime.strptime(start_s, "%Y-%m-%d")).days
            if days < 330:
                continue
            # Quedarse con la presentación más reciente por cada cierre anual
            if end_s not in by_end or filed_s > by_end[end_s][0]:
                by_end[end_s] = (filed_s, val)
        if len(by_end) >= 4:
            break

    if len(by_end) < 4:
        return None

    sorted_ends = sorted(by_end.keys(), reverse=True)
    end_new, end_old = sorted_ends[0], sorted_ends[3]
    eps_new, eps_old = by_end[end_new][1], by_end[end_old][1]

    # El CAGR solo tiene sentido si ambos EPS son positivos
    if eps_new <= 0 or eps_old <= 0:
        return None

    years = (datetime.strptime(end_new, "%Y-%m-%d") - datetime.strptime(end_old, "%Y-%m-%d")).days / 365.25
    if years < 2.5:
        return None

    return round((eps_new / eps_old) ** (1 / years) - 1, 4)


def get_market_metrics(ticker: str, sp500_ret, eps_cagr_3y):
    """Obtiene Precio, Market Cap, P/E, Forward P/E, Beta y PEG vía yfinance."""
    tk = yf.Ticker(ticker)
    info = tk.info or {}

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    market_cap = info.get("marketCap")
    pe_ttm = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    beta = info.get("beta")

    # Respaldo de Beta exacto (Cov / Var a 5 años mensual vs S&P 500)
    if beta is None and sp500_ret is not None:
        stock_hist = tk.history(period="5y", interval="1mo")["Close"]
        stock_ret = stock_hist.pct_change().dropna()
        aligned = stock_ret.to_frame("stock").join(sp500_ret.to_frame("mkt"), how="inner")
        if len(aligned) >= 24:
            var_mkt = aligned["mkt"].var()
            if var_mkt and var_mkt != 0:
                beta = aligned["stock"].cov(aligned["mkt"]) / var_mkt

    # PEG: prioriza consenso de Yahoo; si viene vacío, usa P/E TTM / (CAGR 3Y * 100)
    peg = info.get("trailingPegRatio") or info.get("pegRatio")
    if peg is None and pe_ttm and eps_cagr_3y and eps_cagr_3y > 0:
        peg = pe_ttm / (eps_cagr_3y * 100)

    def r(val, dec=2):
        return round(val, dec) if isinstance(val, (int, float)) else None

    return {
        "price": r(price, 2),
        "market_cap": market_cap,
        "pe_ttm": r(pe_ttm, 2),
        "forward_pe": r(forward_pe, 2),
        "eps_cagr_3y": eps_cagr_3y,
        "peg": r(peg, 2),
        "beta": r(beta, 2),
    }

