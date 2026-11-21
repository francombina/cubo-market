import json
import math
import os
import time
from datetime import datetime, timezone

import requests
import yfinance as yf

# Reemplazá por tu Gmail real
HEADERS = {"User-Agent": "Fran Combina francisco.combina@gmail.com"}

TICKERS_FILE = "tickers.json"
OUTPUT_FILE = "data/fundamentales.json"


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


def extract_latest_fy(
    facts: dict,
    tag_specs: list,
    unit: str = "USD",
    is_duration: bool = True,
    target_end: str = None,
    max_date_diff_days: int = 15,
):
    """
    Busca el dato anual (10-K) más reciente entre tags alternativos.
    Si se pasa target_end, exige que el cierre coincida con ese balance
    para no mezclar tags viejos discontinuados.
    """
    all_facts = facts.get("facts", {})
    candidates = []
    target_dt = datetime.strptime(target_end, "%Y-%m-%d") if target_end else None

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

            end_dt = datetime.strptime(end_str, "%Y-%m-%d")

            if target_dt is not None:
                if abs((end_dt - target_dt).days) > max_date_diff_days:
                    continue

            if is_duration:
                start_str = entry.get("start")
                if not start_str:
                    continue
                start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                if (end_dt - start_dt).days < 330:
                    continue

            candidates.append((end_str, filed_str, -priority, val))

    if not candidates:
        return None, None

    best = max(candidates)
    return clean_num(best[3]), best[0]


def extract_eps_cagr_2y(facts: dict):
    """
    Calcula el CAGR a 2 años del EPS diluido usando los 3 años comparativos
    del MISMO último 10-K (evita distorsiones por stock splits).
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
        entries = us_gaap.get(tag, {}).get("units", {}).get("USD/shares", [])
        valid = []
        for entry in entries:
            if entry.get("form") not in ("10-K", "10-K/A"):
                continue
            start_s, end_s, filed_s, accn, val = (
                entry.get("start"),
                entry.get("end"),
                entry.get("filed"),
                entry.get("accn"),
                entry.get("val"),
            )
            if not (start_s and end_s and filed_s and accn) or val is None:
                continue
            days = (
                datetime.strptime(end_s, "%Y-%m-%d")
                - datetime.strptime(start_s, "%Y-%m-%d")
            ).days
            if days < 330:
                continue
            valid.append((end_s, filed_s, accn, float(val)))

        if not valid:
            continue

        # Tomar el accession number (accn) del 10-K más reciente
        latest_accn = max(valid, key=lambda x: (x[0], x[1]))[2]
        same_filing = {}
        for end_s, filed_s, accn, val in valid:
            if accn == latest_accn:
                same_filing[end_s] = val

        if len(same_filing) < 3:
            continue

        sorted_ends = sorted(same_filing.keys(), reverse=True)
        end_new, end_old = sorted_ends[0], sorted_ends[2]
        eps_new, eps_old = same_filing[end_new], same_filing[end_old]

        if eps_new <= 0 or eps_old <= 0:
            return None

        years = (
            datetime.strptime(end_new, "%Y-%m-%d")
            - datetime.strptime(end_old, "%Y-%m-%d")
        ).days / 365.25
        if years < 1.5:
            return None

        return clean_num((eps_new / eps_old) ** (1 / years) - 1, 4)

    return None


def get_market_metrics(ticker: str, sp500_ret, eps_cagr_2y, fcf):
    """Obtiene datos de mercado y múltiplos vía yfinance con respaldos propios."""
    empty = {
        "price": None,
        "market_cap": None,
        "pe_ttm": None,
        "forward_pe": None,
        "peg": None,
        "beta": None,
        "price_to_fcf": None,
        "ev_to_ebitda": None,
        "yf_shares": None,
        "yf_eps": None,
    }

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        price = clean_num(info.get("currentPrice") or info.get("regularMarketPrice"), 2)
        market_cap = clean_num(info.get("marketCap"), 0)
        pe_ttm = clean_num(info.get("trailingPE"), 2)
        forward_pe = clean_num(info.get("forwardPE"), 2)
        beta = clean_num(info.get("beta"), 2)
        ev_to_ebitda = clean_num(info.get("enterpriseToEbitda"), 2)
        yf_shares = clean_num(info.get("sharesOutstanding"), 0)
        yf_eps = clean_num(info.get("trailingEps"), 2)

        # Respaldo matemático de Beta (5 años mensual vs S&P 500)
        if beta is None and sp500_ret is not None:
            hist = tk.history(period="5y", interval="1mo")["Close"]
            stock_ret = hist.pct_change().dropna()
            if not stock_ret.empty:
                stock_ret.index = stock_ret.index.tz_localize(None).to_period("M")
                aligned = stock_ret.to_frame("stock").join(
                    sp500_ret.to_frame("mkt"), how="inner"
                )
                if len(aligned) >= 24:
                    var_mkt = aligned["mkt"].var()
                    if var_mkt and var_mkt != 0:
                        beta = clean_num(aligned["stock"].cov(aligned["mkt"]) / var_mkt, 2)

        # PEG: prioriza consenso de Yahoo; si es null usa P/E TTM / (EPS CAGR * 100)
        peg = clean_num(info.get("trailingPegRatio") or info.get("pegRatio"), 2)
        if peg is None and pe_ttm and pe_ttm > 0 and eps_cagr_2y and eps_cagr_2y > 0:
            peg = clean_num(pe_ttm / (eps_cagr_2y * 100), 2)

        # Price / Free Cash Flow
        price_to_fcf = None
        if market_cap and fcf and fcf > 0:
            price_to_fcf = safe_div(market_cap, fcf, 2)

        return {
            "price": price,
            "market_cap": market_cap,
            "pe_ttm": pe_ttm,
            "forward_pe": forward_pe,
            "peg": peg,
            "beta": beta,
            "price_to_fcf": price_to_fcf,
            "ev_to_ebitda": ev_to_ebitda,
            "yf_shares": yf_shares,
            "yf_eps": yf_eps,
        }
    except Exception as e:
        print(f"AVISO yfinance en {ticker}: {e}")
        return empty


def parse_company(ticker: str, cik: str, sp500_ret) -> dict:
    facts = get_company_facts(cik)

    # 1. Determinar fecha ancla del último 10-K usando Revenue o Net Income
    revenue, rev_end = extract_latest_fy(
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

    net_income, ni_end = extract_latest_fy(
        facts,
        [("us-gaap", "NetIncomeLoss")],
        is_duration=True,
    )

    fy_end = rev_end or ni_end

    # Si revenue y net_income vinieron de años distintos, alinear al más reciente
    if rev_end and ni_end and rev_end != ni_end:
        fy_end = max(rev_end, ni_end)
        revenue, _ = extract_latest_fy(
            facts,
            [
                ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                ("us-gaap", "Revenues"),
                ("us-gaap", "SalesRevenueNet"),
                ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
                ("us-gaap", "InterestAndDividendIncomeOperating"),
            ],
            is_duration=True,
            target_end=fy_end,
        )
        net_income, _ = extract_latest_fy(
            facts,
            [("us-gaap", "NetIncomeLoss")],
            is_duration=True,
            target_end=fy_end,
        )

    # 2. Extraer el resto de las cuentas ancladas a ese mismo fy_end
    op_income, _ = extract_latest_fy(
        facts,
        [("us-gaap", "OperatingIncomeLoss")],
        is_duration=True,
        target_end=fy_end,
    )

    eps_diluted, _ = extract_latest_fy(
        facts,
        [
            ("us-gaap", "EarningsPerShareDiluted"),
            ("us-gaap", "EarningsPerShareBasic"),
        ],
        unit="USD/shares",
        is_duration=True,
        target_end=fy_end,
    )

    assets, _ = extract_latest_fy(
        facts,
        [("us-gaap", "Assets")],
        is_duration=False,
        target_end=fy_end,
    )

    equity, _ = extract_latest_fy(
        facts,
        [
            ("us-gaap", "StockholdersEquity"),
            (
                "us-gaap",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            ),
        ],
        is_duration=False,
        target_end=fy_end,
    )

    long_term_debt, _ = extract_latest_fy(
        facts,
        [
            ("us-gaap", "LongTermDebtNoncurrent"),
            ("us-gaap", "LongTermDebt"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
        ],
        is_duration=False,
        target_end=fy_end,
    )

    cash, _ = extract_latest_fy(
        facts,
        [
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
            (
                "us-gaap",
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            ),
        ],
        is_duration=False,
        target_end=fy_end,
    )

    ocf, _ = extract_latest_fy(
        facts,
        [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
        is_duration=True,
        target_end=fy_end,
    )

    # Tags ampliados de CapEx (cubre AMZN, NVDA, V, LLY, etc.)
    capex, _ = extract_latest_fy(
        facts,
        [
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
            ("us-gaap", "PaymentsToAcquireProductiveAssets"),
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"),
            ("us-gaap", "PaymentsForCapitalImprovements"),
        ],
        is_duration=True,
        target_end=fy_end,
    )

    sec_shares, _ = extract_latest_fy(
        facts,
        [
            ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
            ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
        ],
        unit="shares",
        is_duration=True,
        target_end=fy_end,
    )
    if sec_shares is None:
        sec_shares, _ = extract_latest_fy(
            facts,
            [
                ("dei", "EntityCommonStockSharesOutstanding"),
                ("us-gaap", "CommonStockSharesOutstanding"),
            ],
            unit="shares",
            is_duration=False,
            target_end=fy_end,
            max_date_diff_days=60,
        )

    fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None
    eps_cagr_2y = extract_eps_cagr_2y(facts)

    # 3. Métricas de mercado vía yfinance
    mkt = get_market_metrics(ticker, sp500_ret, eps_cagr_2y, fcf)

    # Para empresas multiclase (BRK-B, META, V), yfinance da el share count alineado al ticker
    shares_final = mkt["yf_shares"] or sec_shares
    eps_final = eps_diluted if eps_diluted is not None else mkt["yf_eps"]

    return {
        "ticker": ticker,
        "cik": cik,
        "name": facts.get("entityName", ""),
        "fy_end": fy_end,
        # Mercado y Valuación
        "price": mkt["price"],
        "market_cap": mkt["market_cap"],
        "pe_ttm": mkt["pe_ttm"],
        "forward_pe": mkt["forward_pe"],
        "eps_cagr_2y": eps_cagr_2y,
        "peg": mkt["peg"],
        "beta": mkt["beta"],
        "price_to_fcf": mkt["price_to_fcf"],
        "ev_to_ebitda": mkt["ev_to_ebitda"],
        # Fundamentales SEC (10-K)
        "revenue": revenue,
        "operating_income": op_income,
        "net_income": net_income,
        "eps_diluted": clean_num(eps_final, 2),
        "assets": assets,
        "equity": equity,
        "long_term_debt": long_term_debt,
        "cash": cash,
        "operating_cash_flow": ocf,
        "capex": capex,
        "free_cash_flow": fcf,
        "shares_outstanding": shares_final,
        # Rentabilidad y Solvencia
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
                f"OK: {ticker} | Price: {company_data['price']} | "
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
