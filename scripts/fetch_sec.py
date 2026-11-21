import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

# Reemplazá por tu Gmail real
HEADERS = {"User-Agent": "fran combina francisco.combina@gmail.com"}

TICKERS_FILE = "tickers.json"
OUTPUT_FILE = "data/fundamentales.json"
HISTORY_YEARS = 10
HISTORY_QUARTERS = 40


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


def adjust_for_splits(val, filed_str: str, splits: list, decimals: int = 4):
    """Ajusta un EPS histórico por los splits ocurridos DESPUÉS de la fecha filed."""
    if val is None:
        return None
    if not filed_str or not splits:
        return clean_num(val, decimals)
    filed_date = datetime.strptime(filed_str, "%Y-%m-%d").date()
    factor = 1.0
    for split_date, ratio in splits:
        if split_date > filed_date and ratio > 0:
            factor *= ratio
    return clean_num(val / factor, decimals)


def extract_duration_buckets(
    facts: dict,
    tag_specs: list,
    unit: str = "USD",
    splits: list = None,
) -> dict:
    """
    Extrae series de cuentas de flujo (10-Q y 10-K) separadas por duración:
    3m (trimestre aislado), 6m (YTD Q2), 9m (YTD Q3) y 12m (año completo 10-K).
    Si se pasa splits (para EPS), ajusta cada valor antes de guardarlo.
    """
    all_facts = facts.get("facts", {})
    raw = {"3m": {}, "6m": {}, "9m": {}, "12m": {}}

    for priority, (taxonomy, tag) in enumerate(tag_specs):
        tag_data = all_facts.get(taxonomy, {}).get(tag, {})
        entries = tag_data.get("units", {}).get(unit, [])

        for entry in entries:
            form = entry.get("form")
            if form not in ("10-Q", "10-Q/A", "10-K", "10-K/A"):
                continue

            start_str = entry.get("start")
            end_str = entry.get("end")
            filed_str = entry.get("filed")
            val = clean_num(entry.get("val"))
            if not (start_str and end_str and filed_str) or val is None:
                continue

            days = (
                datetime.strptime(end_str, "%Y-%m-%d")
                - datetime.strptime(start_str, "%Y-%m-%d")
            ).days

            bucket = None
            if 70 <= days <= 115:
                bucket = "3m"
            elif 155 <= days <= 205 and form in ("10-Q", "10-Q/A"):
                bucket = "6m"
            elif 245 <= days <= 295 and form in ("10-Q", "10-Q/A"):
                bucket = "9m"
            elif 330 <= days <= 400 and form in ("10-K", "10-K/A"):
                bucket = "12m"

            if not bucket:
                continue

            if splits is not None:
                val = adjust_for_splits(val, filed_str, splits, decimals=4)
                if val is None:
                    continue

            rank = (filed_str, -priority)
            if end_str not in raw[bucket] or rank > raw[bucket][end_str][0]:
                raw[bucket][end_str] = (rank, val, filed_str)

    return {
        b: {end_s: (val, filed_s) for end_s, (_, val, filed_s) in d.items()}
        for b, d in raw.items()
    }


def extract_instant_series(
    facts: dict,
    tag_specs: list,
    unit: str = "USD",
) -> dict:
    """Extrae cuentas patrimoniales (foto al cierre) de todos los 10-Q y 10-K."""
    all_facts = facts.get("facts", {})
    by_end = {}

    for priority, (taxonomy, tag) in enumerate(tag_specs):
        tag_data = all_facts.get(taxonomy, {}).get(tag, {})
        entries = tag_data.get("units", {}).get(unit, [])

        for entry in entries:
            if entry.get("form") not in ("10-Q", "10-Q/A", "10-K", "10-K/A"):
                continue

            end_str = entry.get("end")
            filed_str = entry.get("filed")
            val = clean_num(entry.get("val"))
            if not end_str or not filed_str or val is None:
                continue

            rank = (filed_str, -priority)
            if end_str not in by_end or rank > by_end[end_str][0]:
                by_end[end_str] = (rank, val, filed_str)

    return {end_s: (val, filed_s) for end_s, (_, val, filed_s) in by_end.items()}


def lookup_near_date(series: dict, target_end: str, max_days: int = 15):
    """Busca un valor en una serie exactamente en target_end o dentro de +/- max_days."""
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


def cluster_dates(date_set: set, prefer_set: set, min_gap_days: int = 20) -> list:
    """Ordena fechas de cierre y agrupa duplicados separados por pocos días."""
    anchors = []
    for d in sorted(date_set):
        dt = datetime.strptime(d, "%Y-%m-%d")
        if anchors:
            prev_dt = datetime.strptime(anchors[-1], "%Y-%m-%d")
            if (dt - prev_dt).days <= min_gap_days:
                if d in prefer_set and anchors[-1] not in prefer_set:
                    anchors[-1] = d
                continue
        anchors.append(d)
    return anchors


def quarter_label(end_str: str) -> str:
    """
    Convierte una fecha de cierre a etiqueta YYYY-Q# (ej. 2026-Q3).
    Si el cierre cae en los primeros 7 días del mes (calendarios 52/53 semanas),
    lo ajusta al mes anterior para alinear empresas entre sí sin colisiones.
    """
    dt = datetime.strptime(end_str, "%Y-%m-%d")
    if dt.day <= 7:
        dt = dt - timedelta(days=10)
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


def resolve_standalone_quarters(buckets: dict, q_anchors: list) -> list:
    """
    Resuelve el dato aislado de cada trimestre:
    - Usa el dato directo de 3 meses si existe (Q1, Q2, Q3 de resultados).
    - Si viene acumulado YTD (flujo de caja) o es Q4 (10-K anual), resta el
      acumulado previo (6m-3m, 9m-6m, 12m-9m) o los 3 trimestres anteriores.
    """
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in q_anchors]
    resolved = [None] * len(q_anchors)

    for i, q_end in enumerate(q_anchors):
        # 1. Dato aislado de 3 meses directo
        val_3m, _ = lookup_near_date(buckets["3m"], q_end)
        if val_3m is not None:
            resolved[i] = val_3m
            continue

        prev_ok = i >= 1 and (65 <= (dts[i] - dts[i - 1]).days <= 125)

        if prev_ok:
            prev_end = q_anchors[i - 1]

            # 2. Q2 desde YTD 6 meses - YTD 3 meses
            val_6m, _ = lookup_near_date(buckets["6m"], q_end)
            prev_3m, _ = lookup_near_date(buckets["3m"], prev_end)
            if val_6m is not None and prev_3m is not None:
                resolved[i] = clean_num(val_6m - prev_3m)
                continue

            # 3. Q3 desde YTD 9 meses - YTD 6 meses
            val_9m, _ = lookup_near_date(buckets["9m"], q_end)
            prev_6m, _ = lookup_near_date(buckets["6m"], prev_end)
            if val_9m is not None and prev_6m is not None:
                resolved[i] = clean_num(val_9m - prev_6m)
                continue

            # 4. Q4 desde Anual 12 meses - YTD 9 meses
            val_12m, _ = lookup_near_date(buckets["12m"], q_end)
            prev_9m, _ = lookup_near_date(buckets["9m"], prev_end)
            if val_12m is not None and prev_9m is not None:
                resolved[i] = clean_num(val_12m - prev_9m)
                continue

        # 5. Respaldo Q4: Anual 12 meses - (Q1 + Q2 + Q3)
        val_12m, _ = lookup_near_date(buckets["12m"], q_end)
        if val_12m is not None and i >= 3:
            if 240 <= (dts[i] - dts[i - 3]).days <= 305:
                prev_three = (resolved[i - 3], resolved[i - 2], resolved[i - 1])
                if all(v is not None for v in prev_three):
                    resolved[i] = clean_num(val_12m - sum(prev_three))

    return resolved


def get_market_data_and_history(ticker: str, sp500_ret):
    """Descarga info actual, precios diarios (ajustados solo por splits) y tabla de splits."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

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
    """Obtiene el precio de cierre en target_end (o último día hábil previo hasta 10 días)."""
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


def parse_company(ticker: str, cik: str, sp500_ret) -> dict:
    facts = get_company_facts(cik)
    mkt = get_market_data_and_history(ticker, sp500_ret)

    # 1. Extraer buckets de duración (3m, 6m, 9m, 12m) y series instantáneas
    rev_b = extract_duration_buckets(
        facts,
        [
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "Revenues"),
            ("us-gaap", "SalesRevenueNet"),
            ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
            ("us-gaap", "InterestAndDividendIncomeOperating"),
        ],
    )
    op_b = extract_duration_buckets(facts, [("us-gaap", "OperatingIncomeLoss")])
    ni_b = extract_duration_buckets(facts, [("us-gaap", "NetIncomeLoss")])
    eps_b = extract_duration_buckets(
        facts,
        [
            ("us-gaap", "EarningsPerShareDiluted"),
            ("us-gaap", "EarningsPerShareBasic"),
        ],
        unit="USD/shares",
        splits=mkt["splits"],
    )
    ocf_b = extract_duration_buckets(
        facts,
        [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
    )
    capex_b = extract_duration_buckets(
        facts,
        [
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
            ("us-gaap", "PaymentsToAcquireProductiveAssets"),
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets"),
            ("us-gaap", "PaymentsForCapitalImprovements"),
        ],
    )
    shares_dur_b = extract_duration_buckets(
        facts,
        [
            ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
            ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
        ],
        unit="shares",
    )

    assets_s = extract_instant_series(facts, [("us-gaap", "Assets")])
    equity_s = extract_instant_series(
        facts,
        [
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ],
    )
    debt_s = extract_instant_series(
        facts,
        [
            ("us-gaap", "LongTermDebtNoncurrent"),
            ("us-gaap", "LongTermDebt"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
        ],
    )
    cash_s = extract_instant_series(
        facts,
        [
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
            ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        ],
    )
    shares_inst_s = extract_instant_series(
        facts,
        [
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
        ],
        unit="shares",
    )

    # =========================================================================
    # 2. HISTORIAL ANUAL (10 AÑOS, 10-K)
    # =========================================================================
    annual_anchors = cluster_dates(
        set(rev_b["12m"].keys()) | set(ni_b["12m"].keys()),
        prefer_set=set(rev_b["12m"].keys()),
    )[-(HISTORY_YEARS + 3):]

    annual_buf = []
    for fy_end in annual_anchors:
        revenue, _ = lookup_near_date(rev_b["12m"], fy_end)
        op_income, _ = lookup_near_date(op_b["12m"], fy_end)
        net_income, _ = lookup_near_date(ni_b["12m"], fy_end)
        adj_eps, _ = lookup_near_date(eps_b["12m"], fy_end)
        assets, _ = lookup_near_date(assets_s, fy_end)
        equity, _ = lookup_near_date(equity_s, fy_end)
        debt, _ = lookup_near_date(debt_s, fy_end)
        cash, _ = lookup_near_date(cash_s, fy_end)
        ocf, _ = lookup_near_date(ocf_b["12m"], fy_end)
        capex, _ = lookup_near_date(capex_b["12m"], fy_end)

        sec_shares, _ = lookup_near_date(shares_dur_b["12m"], fy_end)
        if sec_shares is None:
            sec_shares, _ = lookup_near_date(shares_inst_s, fy_end, max_days=60)

        fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None
        price_at_end = get_price_at_date(mkt["close_series"], fy_end)
        hist_pe = clean_num(price_at_end / adj_eps, 2) if (price_at_end and adj_eps and adj_eps > 0) else None

        annual_buf.append(
            {
                "year": int(fy_end[:4]),
                "fy_end": fy_end,
                "price": price_at_end,
                "revenue": revenue,
                "operating_income": op_income,
                "net_income": net_income,
                "eps_diluted": clean_num(adj_eps, 2),
                "eps_raw": adj_eps,
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

    for i, row in enumerate(annual_buf):
        cagr = None
        for lookback in (3, 2):
            if i >= lookback:
                eps_now = row["eps_raw"]
                eps_prev = annual_buf[i - lookback]["eps_raw"]
                if eps_now and eps_prev and eps_now > 0 and eps_prev > 0:
                    days = (
                        datetime.strptime(row["fy_end"], "%Y-%m-%d")
                        - datetime.strptime(annual_buf[i - lookback]["fy_end"], "%Y-%m-%d")
                    ).days
                    years = days / 365.25
                    if years >= lookback - 0.3:
                        cagr = clean_num((eps_now / eps_prev) ** (1 / years) - 1, 4)
                        break

        row["eps_cagr"] = cagr
        row["peg"] = (
            clean_num(row["pe"] / (cagr * 100), 2)
            if (row["pe"] and row["pe"] > 0 and cagr is not None and cagr >= 0.01)
            else None
        )

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
        for r in annual_buf[-HISTORY_YEARS:]
    ]

    # =========================================================================
    # 3. HISTORIAL TRIMESTRAL (HASTA 40 TRIMESTRES, 10-Q + Q4 DERIVADO)
    # =========================================================================
    q_dates_all = (
        set(rev_b["3m"].keys())
        | set(ni_b["3m"].keys())
        | set(rev_b["12m"].keys())
        | set(ni_b["12m"].keys())
    )
    q_prefer = set(rev_b["3m"].keys()) | set(rev_b["12m"].keys())
    # Buffer extra (16 trimestres = 4 años) para calcular TTM + CAGR 3A desde el primer punto
    q_anchors = cluster_dates(q_dates_all, prefer_set=q_prefer)[-(HISTORY_QUARTERS + 16):]
    q_dts = [datetime.strptime(d, "%Y-%m-%d") for d in q_anchors]

    q_rev = resolve_standalone_quarters(rev_b, q_anchors)
    q_op = resolve_standalone_quarters(op_b, q_anchors)
    q_ni = resolve_standalone_quarters(ni_b, q_anchors)
    q_eps = resolve_standalone_quarters(eps_b, q_anchors)
    q_ocf = resolve_standalone_quarters(ocf_b, q_anchors)
    q_capex = resolve_standalone_quarters(capex_b, q_anchors)

    q_buf = []
    for i, q_end in enumerate(q_anchors):
        rev = q_rev[i]
        op = q_op[i]
        ni = q_ni[i]
        eps = q_eps[i]
        ocf = q_ocf[i]
        capex = q_capex[i]
        fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None

        assets, _ = lookup_near_date(assets_s, q_end)
        equity, _ = lookup_near_date(equity_s, q_end)
        debt, _ = lookup_near_date(debt_s, q_end)
        price_q = get_price_at_date(mkt["close_series"], q_end)

        # Verificar si los últimos 4 trimestres son consecutivos para calcular TTM
        consec_4 = i >= 3 and all(
            65 <= (q_dts[k] - q_dts[k - 1]).days <= 125 for k in range(i - 2, i + 1)
        )

        ttm_eps = None
        ttm_ni = None
        if consec_4:
            eps_4 = q_eps[i - 3 : i + 1]
            if all(v is not None for v in eps_4):
                ttm_eps = sum(eps_4)
            ni_4 = q_ni[i - 3 : i + 1]
            if all(v is not None for v in ni_4):
                ttm_ni = sum(ni_4)

        pe_q = clean_num(price_q / ttm_eps, 2) if (price_q and ttm_eps and ttm_eps > 0) else None
        roe_q = safe_div(ttm_ni, equity)
        roa_q = safe_div(ttm_ni, assets)

        q_buf.append(
            {
                "period": quarter_label(q_end),
                "fy_end": q_end,
                "price": price_q,
                "revenue": clean_num(rev, 0),
                "operating_income": clean_num(op, 0),
                "net_income": clean_num(ni, 0),
                "eps_diluted": clean_num(eps, 2),
                "free_cash_flow": clean_num(fcf, 0),
                "ttm_eps": ttm_eps,
                "pe": pe_q,
                "roe": roe_q,
                "roa": roa_q,
                "operating_margin": safe_div(op, rev),
                "net_margin": safe_div(ni, rev),
                "debt_to_equity": safe_div(debt, equity),
            }
        )

    # Calcular CAGR TTM (3 años = 12 trimestres, fallback 2 años = 8 trimestres) y PEG trimestral
    for i, row in enumerate(q_buf):
        cagr = None
        for lookback_q in (12, 8):
            if i >= lookback_q:
                eps_now = row["ttm_eps"]
                eps_prev = q_buf[i - lookback_q]["ttm_eps"]
                if eps_now and eps_prev and eps_now > 0 and eps_prev > 0:
                    years = (q_dts[i] - q_dts[i - lookback_q]).days / 365.25
                    target_y = lookback_q / 4.0
                    if abs(years - target_y) <= 0.35:
                        cagr = clean_num((eps_now / eps_prev) ** (1 / years) - 1, 4)
                        break

        row["eps_cagr"] = cagr
        row["peg"] = (
            clean_num(row["pe"] / (cagr * 100), 2)
            if (row["pe"] and row["pe"] > 0 and cagr is not None and cagr >= 0.01)
            else None
        )

    # Deduplicar por etiqueta period (por si una empresa cambió su calendario fiscal)
    q_by_period = {}
    for r in q_buf[-HISTORY_QUARTERS:]:
        q_by_period[r["period"]] = {
            "period": r["period"],
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
    quarterly_history = list(q_by_period.values())

    # =========================================================================
    # 4. RESUMEN ACTUAL (100% COMPATIBLE CON GOOGLE SHEETS)
    # =========================================================================
    latest = annual_buf[-1] if annual_buf else {}
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
        "price": mkt["price"],
        "market_cap": mkt["market_cap"],
        "pe_ttm": pe_ttm,
        "forward_pe": mkt["forward_pe"],
        "eps_cagr_2y": latest_cagr,
        "peg": peg_current,
        "beta": mkt["beta"],
        "price_to_fcf": price_to_fcf,
        "ev_to_ebitda": mkt["ev_to_ebitda"],
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
        "operating_margin": latest.get("operating_margin"),
        "net_margin": latest.get("net_margin"),
        "roe": latest.get("roe"),
        "roa": latest.get("roa"),
        "debt_to_equity": latest.get("debt_to_equity"),
        "history": history,
        "quarterly_history": quarterly_history,
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
                f"OK: {ticker} | Anual: {len(company_data['history'])}A | "
                f"Trim: {len(company_data['quarterly_history'])}T | "
                f"PE: {company_data['pe_ttm']} | PEG: {company_data['peg']}"
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
