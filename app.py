"""Interactive dashboard: sweet spot between purchase price and rent.

Goal
- Find the sweet spot between purchase price and monthly rent.
- Show capital development via amortization (and property equity).

Scope
- No "potential levers" or market assumptions.
- Parameters are editable; Kaufpreis/Miete can be adjusted quickly.

Notes
- This is a simplified model (no operating costs, vacancy). Tax/valuation options are simplified.

Run
    streamlit run app.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def eur(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    return f"{x:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    return f"{x*100:.2f}%".replace(".", ",")


def safe_div(a: float, b: float) -> float:
    return float("nan") if b == 0 else a / b


def annuity_month(principal: float, interest: float, initial_repayment: float) -> float:
    """Monthly annuity payment using German 'initial repayment' convention.

    Payment is assumed constant over time (annuity loan):
      payment = principal * (interest + initial_repayment) / 12
    """
    return principal * (interest + initial_repayment) / 12.0


def amortization_schedule(
    *,
    principal: float,
    interest: float,
    initial_repayment: float,
    property_value: float,
    appreciation_rate: float,
    years: int,
) -> pd.DataFrame:
    payment = annuity_month(principal, interest, initial_repayment)
    months = years * 12

    remaining = principal
    rows: list[dict[str, float]] = []

    for m in range(months + 1):
        year = m / 12.0
        current_property_value = property_value * ((1.0 + appreciation_rate) ** year)
        amortized_equity = principal - remaining
        property_equity = current_property_value - remaining
        rows.append(
            {
                "year": year,
                "remaining_debt": remaining,
                "amortized_equity": amortized_equity,
                "property_equity": property_equity,
                "property_value": current_property_value,
                "payment": payment,
            }
        )

        if m == months:
            break

        interest_payment = remaining * interest / 12.0
        principal_payment = payment - interest_payment
        if principal_payment <= 0:
            # Non-amortizing (or negative amortization) edge case.
            # Keep remaining constant to avoid going negative.
            continue

        remaining = max(0.0, remaining - principal_payment)
        if remaining == 0.0:
            # Fully repaid early: pad remaining months with zeros.
            for mm in range(m + 1, months + 1):
                year2 = mm / 12.0
                current_property_value2 = property_value * ((1.0 + appreciation_rate) ** year2)
                rows.append(
                    {
                        "year": year2,
                        "remaining_debt": 0.0,
                        "amortized_equity": principal,
                        "property_equity": current_property_value2,
                        "property_value": current_property_value2,
                        "payment": payment,
                    }
                )
            break

    return pd.DataFrame(rows)


def amortization_schedule_detailed(
    *,
    principal: float,
    interest: float,
    initial_repayment: float,
    years: int,
) -> pd.DataFrame:
    """Monthly amortization schedule including interest/principal payments."""
    payment = annuity_month(principal, interest, initial_repayment)
    months = int(years) * 12

    remaining = float(principal)
    rows: list[dict[str, float]] = []

    for m in range(1, months + 1):
        interest_payment = remaining * float(interest) / 12.0
        principal_payment = payment - interest_payment
        if principal_payment < 0:
            principal_payment = 0.0
        remaining = max(0.0, remaining - principal_payment)

        rows.append(
            {
                "month": float(m),
                "year": math.ceil(m / 12.0),
                "payment": float(payment),
                "interest_payment": float(interest_payment),
                "principal_payment": float(principal_payment),
                "remaining_debt": float(remaining),
            }
        )
        if remaining <= 1e-9:
            # Fully repaid: stop early
            break

    return pd.DataFrame(rows)


def aggregate_yearly_amortization(monthly: pd.DataFrame, years: int) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame(
            {
                "year": [float(y) for y in range(1, years + 1)],
                "interest": [0.0] * years,
                "principal": [0.0] * years,
                "payment": [0.0] * years,
            }
        )

    g = (
        monthly.groupby("year", as_index=False)
        .agg(
            interest=("interest_payment", "sum"),
            principal=("principal_payment", "sum"),
            payment=("payment", "sum"),
        )
        .astype({"year": float})
    )

    # Pad missing years
    have = set(int(x) for x in g["year"].tolist())
    rows: list[dict[str, float]] = g.to_dict(orient="records")
    for y in range(1, years + 1):
        if y not in have:
            rows.append({"year": float(y), "interest": 0.0, "principal": 0.0, "payment": 0.0})
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return out


@dataclass(frozen=True)
class PurchaseCosts:
    broker_rate: float
    grunderwerbsteuer_rate: float
    notar_rate: float
    grundbuch_rate: float

    def total_rate(self) -> float:
        return self.broker_rate + self.grunderwerbsteuer_rate + self.notar_rate + self.grundbuch_rate


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def depreciation_linear(*, basis: float, rate: float, years: int) -> list[float]:
    """Simple linear depreciation: constant amount per year.

    Note: This is a simplified engine. German AfA has more nuances (e.g. pro-rata in year of acquisition).
    """
    basis = float(max(0.0, basis))
    rate = float(max(0.0, rate))
    if years <= 0 or basis == 0.0 or rate == 0.0:
        return [0.0 for _ in range(max(0, years))]

    annual = basis * rate
    out: list[float] = []
    remaining = basis
    for _ in range(years):
        a = min(annual, remaining)
        out.append(a)
        remaining -= a
        if remaining <= 1e-9:
            out.extend([0.0 for _ in range(years - len(out))])
            break
    return out


def depreciation_schedule_percent(*, basis: float, percents: list[float], years: int) -> list[float]:
    """Percent schedule where each year uses a percent of the original basis.

    percents are provided as fractions (e.g. 0.09 for 9%).
    """
    basis = float(max(0.0, basis))
    years = int(max(0, years))
    out: list[float] = []
    remaining = basis

    for i in range(years):
        p = float(percents[i]) if i < len(percents) else 0.0
        p = max(0.0, p)
        a = min(basis * p, remaining)
        out.append(a)
        remaining -= a
        if remaining <= 1e-9:
            out.extend([0.0 for _ in range(years - len(out))])
            break
    return out


def shift_schedule(values: list[float], years: int, start_year: int) -> list[float]:
    """Shift a schedule so that it starts at start_year (1-based)."""
    years = int(max(0, years))
    start_year = int(max(1, start_year))
    out = [0.0 for _ in range(years)]
    for i, v in enumerate(values):
        y = start_year - 1 + i
        if 0 <= y < years:
            out[y] = float(v)
    return out


def prorate_year1(values: list[float], acquisition_month: int) -> list[float]:
    """Prorate year 1 amount by remaining months from acquisition_month..12."""
    if not values:
        return values
    m = int(acquisition_month)
    m = min(12, max(1, m))
    factor = (13 - m) / 12.0
    out = list(values)
    out[0] = float(out[0]) * float(factor)
    return out


def sanitize_percent_list(raw: str) -> list[float]:
    """Parse a user string like '9,9,9,9,9,9,9,9,7,7,7,7' into fractions."""
    s = (raw or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        p2 = p.replace("%", "").replace(" ", "").replace(",", ".")
        try:
            val = float(p2) / 100.0
        except ValueError:
            continue
        out.append(max(0.0, val))
    return out


def compute_tax_cashflow_de(
    *,
    years: int,
    rent_year: float,
    interest_year: float,
    operating_costs_year: float,
    depreciation_year: list[float],
    marginal_tax_rate: float,
    income_cap_year: float | None,
) -> pd.DataFrame:
    """Compute taxable result and after-tax cashflow.

    Cashflow definition here (simplified):
      before_tax = rent - interest - operating_costs
      taxable = before_tax - depreciation
      tax = taxable * tax_rate (negative taxable => tax refund)
      after_tax = before_tax - tax

    If income_cap_year is set, we cap the loss offset to that amount.
    """
    years = int(max(1, years))
    tax_rate = _clamp01(marginal_tax_rate)
    rent_year = float(rent_year)
    interest_year = float(interest_year)
    operating_costs_year = float(operating_costs_year)

    rows: list[dict[str, float]] = []
    cap = None
    if income_cap_year is not None and float(income_cap_year) > 0:
        cap = float(income_cap_year)

    for y in range(1, years + 1):
        dep = float(depreciation_year[y - 1]) if y - 1 < len(depreciation_year) else 0.0
        before_tax = rent_year - interest_year - operating_costs_year
        taxable = before_tax - dep

        # Cap loss offset (very simplified; real DE tax law has additional constraints)
        effective_taxable = taxable
        if cap is not None:
            if effective_taxable < 0:
                effective_taxable = -min(abs(effective_taxable), cap)
            else:
                effective_taxable = min(effective_taxable, cap)

        tax = effective_taxable * tax_rate
        after_tax = before_tax - tax

        rows.append(
            {
                "year": float(y),
                "rent": rent_year,
                "interest": interest_year,
                "operating_costs": operating_costs_year,
                "depreciation": dep,
                "before_tax": before_tax,
                "taxable": taxable,
                "tax": tax,
                "after_tax": after_tax,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Immobilien-Kennzahlen", layout="wide")

    DEFAULT_PRICE = 1_190_000.0
    DEFAULT_RENT = 3_626.0
    DEFAULT_INTEREST_PCT = 4.0
    DEFAULT_TILGUNG_MIN_PCT = 1.0
    DEFAULT_TILGUNG_MAX_PCT = 1.5
    DEFAULT_LIVING_AREA = 370.0
    DEFAULT_COMMERCIAL_AREA = 150.0

    st.title("Sweet Spot: Kaufpreis ↔ Miete (Kapitalentwicklung durch Tilgung)")
    st.caption(
        "Vereinfachtes Modell ohne Kosten/Steuern: du kannst Kaufpreis, Miete, Zins, Tilgung (Min/Max), Flächen und optional das Darlehen direkt als Werte ändern."
    )

    with st.sidebar:
        st.header("Inputs")
        # --- Kaufpreis: Slider + Direkteingabe (synchron) ---
        if "purchase_price_slider" not in st.session_state:
            st.session_state.purchase_price_slider = int(DEFAULT_PRICE)
        if "purchase_price_num" not in st.session_state:
            st.session_state.purchase_price_num = float(DEFAULT_PRICE)

        def _sync_price_from_slider() -> None:
            st.session_state.purchase_price_num = float(st.session_state.purchase_price_slider)

        def _sync_price_from_num() -> None:
            v = float(st.session_state.purchase_price_num)
            v = min(3_000_000.0, max(100_000.0, v))
            # round to slider step
            v = round(v / 10_000.0) * 10_000.0
            st.session_state.purchase_price_num = v
            st.session_state.purchase_price_slider = int(v)

        st.slider(
            "Kaufpreis (EUR)",
            min_value=100_000,
            max_value=3_000_000,
            step=10_000,
            key="purchase_price_slider",
            on_change=_sync_price_from_slider,
        )
        st.number_input(
            "Kaufpreis (direkt) (EUR)",
            min_value=100_000.0,
            max_value=3_000_000.0,
            step=10_000.0,
            key="purchase_price_num",
            on_change=_sync_price_from_num,
        )

        # --- Miete: Slider + Direkteingabe (synchron) ---
        if "rent_slider" not in st.session_state:
            st.session_state.rent_slider = int(DEFAULT_RENT)
        if "rent_num" not in st.session_state:
            st.session_state.rent_num = float(DEFAULT_RENT)

        def _sync_rent_from_slider() -> None:
            st.session_state.rent_num = float(st.session_state.rent_slider)

        def _sync_rent_from_num() -> None:
            v = float(st.session_state.rent_num)
            v = min(12_000.0, max(0.0, v))
            v = round(v / 50.0) * 50.0
            st.session_state.rent_num = v
            st.session_state.rent_slider = int(v)

        st.subheader("Flächen")
        living_area = st.number_input("Wohnfläche (m²)", min_value=0.0, value=DEFAULT_LIVING_AREA, step=5.0)
        has_commercial = st.toggle("Gewerbefläche vorhanden", value=True)
        commercial_area = 0.0
        if has_commercial:
            commercial_area = st.number_input("Gewerbefläche (m²)", min_value=0.0, value=DEFAULT_COMMERCIAL_AREA, step=5.0)
        total_area = float(living_area) + float(commercial_area)
        st.caption(f"Gesamtfläche (berechnet): {total_area:.0f} m²")

        st.subheader("Mieteingabe")
        rent_mode = st.radio(
            "Modus",
            options=["Gesamtmiete", "€/m² (Wohnen/Gewerbe)"],
            index=0,
            key="rent_mode_ui",
            help="Wenn du €/m² wählst, wird die Gesamtmiete aus Flächen × €/m² berechnet.",
        )
        # Keep a simple internal flag to avoid string comparisons everywhere
        st.session_state.rent_mode = "sqm" if rent_mode.startswith("€/m²") else "total"

        if st.session_state.rent_mode == "total":
            st.slider(
                "Kaltmiete (EUR/Monat)",
                min_value=0,
                max_value=12_000,
                step=50,
                key="rent_slider",
                on_change=_sync_rent_from_slider,
            )
            st.number_input(
                "Kaltmiete (direkt) (EUR/Monat)",
                min_value=0.0,
                max_value=12_000.0,
                step=50.0,
                key="rent_num",
                on_change=_sync_rent_from_num,
            )

        if st.session_state.rent_mode == "sqm":
            # Initialize defaults to keep consistency with the current total rent without inventing a split.
            if "rent_res_sqm" not in st.session_state or "rent_com_sqm" not in st.session_state:
                fallback = safe_div(float(st.session_state.rent_num), total_area) if total_area > 0 else 0.0
                st.session_state.rent_res_sqm = float(fallback)
                st.session_state.rent_com_sqm = float(fallback)

            rent_res_sqm = st.number_input(
                "Wohnmiete (EUR/m²/Monat)",
                min_value=0.0,
                value=float(st.session_state.rent_res_sqm),
                step=0.1,
                format="%.2f",
                key="rent_res_sqm",
            )
            rent_com_sqm = 0.0
            if has_commercial:
                rent_com_sqm = st.number_input(
                    "Gewerbemiete (EUR/m²/Monat)",
                    min_value=0.0,
                    value=float(st.session_state.rent_com_sqm),
                    step=0.1,
                    format="%.2f",
                    key="rent_com_sqm",
                )

            rent_total_calc = float(living_area) * float(rent_res_sqm) + float(commercial_area) * float(rent_com_sqm)
            # Do NOT write into widget keys (rent_num/rent_slider) here; Streamlit forbids that.
            st.session_state.rent_total_calc = float(rent_total_calc)
            st.info(f"Berechnete Gesamtmiete: {eur(rent_total_calc)} / Monat")

        st.subheader("Finanzierung")
        interest_pct = st.number_input(
            "Sollzins p.a. (%)",
            min_value=0.0,
            max_value=25.0,
            value=DEFAULT_INTEREST_PCT,
            step=0.05,
            format="%.2f",
        )
        tilgung_min_pct = st.number_input(
            "Tilgung min p.a. (%)",
            min_value=0.0,
            max_value=25.0,
            value=DEFAULT_TILGUNG_MIN_PCT,
            step=0.05,
            format="%.2f",
        )
        tilgung_max_pct = st.number_input(
            "Tilgung max p.a. (%)",
            min_value=0.0,
            max_value=25.0,
            value=DEFAULT_TILGUNG_MAX_PCT,
            step=0.05,
            format="%.2f",
        )
        loan = st.number_input(
            "Darlehen (EUR)",
            min_value=0.0,
            value=float(st.session_state.purchase_price_num),
            step=10_000.0,
            help="Default = Kaufpreis. Wenn kleiner, entspricht das Eigenkapital im Objekt (vereinfachend).",
        )

        st.subheader("Erweiterte Optionen")
        appreciation_enabled = st.toggle("Wertsteigerung aktivieren", value=False)
        appreciation_pct = st.number_input(
            "Wertsteigerung p.a. (%)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            disabled=not appreciation_enabled,
        )

        st.subheader("Steuern / AfA (DE)")
        tax_mode: Literal["Aus", "Einfach", "Erweitert (DE)"] = st.radio(
            "Modus",
            options=["Aus", "Einfach", "Erweitert (DE)"],
            index=0,
            key="tax_mode_ui",
            help="Hinweis: Das ist ein Rechenmodell (keine Steuerberatung). Für Sonderfälle bitte mit Steuerberater abgleichen.",
        )

        # --- simple mode: keep previous functionality ---
        afa_in_cashflow = False
        building_share_pct = 0.0
        afa_rate_pct = 0.0
        tax_rate_pct = 0.0
        gross_income_year = 0.0

        # --- extended mode inputs ---
        de_tax_enabled = tax_mode != "Aus"
        de_tax_extended = tax_mode == "Erweitert (DE)"

        if tax_mode == "Einfach":
            st.caption("Einfach: AfA-Basis = Kaufpreis × Gebäudeanteil; lineare AfA; Steuerwirkung über Grenzsteuersatz.")
            afa_in_cashflow = st.toggle("Steuerwirkung in Cashflow einrechnen", value=False)
            building_share_pct = st.number_input(
                "Gebäudeanteil am Kaufpreis (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                format="%.1f",
                help="Vereinfachung: AfA-Basis = Kaufpreis × Gebäudeanteil.",
            )
            afa_rate_pct = st.number_input(
                "AfA-Satz p.a. (%)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                format="%.2f",
            )
            tax_rate_pct = st.number_input(
                "Grenzsteuersatz (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                format="%.1f",
            )
            gross_income_year = st.number_input(
                "Einkommen-Cap p.a. (EUR) (optional)",
                min_value=0.0,
                value=0.0,
                step=1_000.0,
                help="Optional: cappt Verlustverrechnung (sehr vereinfacht). 0 = kein Cap.",
            )

        # Extended DE mode: allocation into land / old building / renovation
        land_share_pct = 0.0
        purchase_costs = PurchaseCosts(0.0, 0.0, 0.0, 0.0)
        operating_costs_year = 0.0
        building_afa_rate_pct = 0.0
        renovation_costs = 0.0
        renovation_afa_scheme: Literal["§7h/§7i (9%×8 + 7%×4)", "Benutzerdefiniert"] = "§7h/§7i (9%×8 + 7%×4)"
        renovation_custom = ""
        renovation_eligible_pct = 100.0
        tax_rate_de_pct = 0.0
        income_cap_de_year = 0.0
        acquisition_month = 1
        renovation_start_year = 1

        if de_tax_extended:
            st.caption(
                "Erweitert: Aufteilung in Grund und Boden (nicht abschreibbar), Altbausubstanz/Gebäude (lineare AfA) und Sanierungskosten (Sanierungs-AfA-Schema)."
            )

            st.subheader("Aufteilung Kauf (Basis)")
            land_share_pct = st.number_input(
                "Anteil Grund & Boden (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                format="%.1f",
                help="Vereinfachte Eingabe. In der Praxis oft über Bodenrichtwert/Arbeitshilfen ermittelt.",
            )

            st.subheader("Kaufnebenkosten (Rate vom Kaufpreis)")
            col_a, col_b = st.columns(2)
            with col_a:
                broker_rate_pct = st.number_input("Makler (%)", min_value=0.0, max_value=20.0, value=3.57, step=0.01, format="%.2f")
                grunderwerbsteuer_pct = st.number_input("GrESt (%)", min_value=0.0, max_value=10.0, value=3.5, step=0.1, format="%.2f")
            with col_b:
                notar_pct = st.number_input("Notar (%)", min_value=0.0, max_value=10.0, value=1.5, step=0.1, format="%.2f")
                grundbuch_pct = st.number_input("Grundbuch (%)", min_value=0.0, max_value=10.0, value=0.5, step=0.1, format="%.2f")
            purchase_costs = PurchaseCosts(
                broker_rate=float(broker_rate_pct) / 100.0,
                grunderwerbsteuer_rate=float(grunderwerbsteuer_pct) / 100.0,
                notar_rate=float(notar_pct) / 100.0,
                grundbuch_rate=float(grundbuch_pct) / 100.0,
            )

            st.subheader("Gebäude-AfA (Altbausubstanz)")
            building_afa_rate_pct = st.number_input(
                "Lineare Gebäude-AfA p.a. (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                format="%.2f",
                help="Vereinfachung: konstante lineare AfA über die Jahre. (z.B. 2% entspricht 50 Jahre).",
            )

            st.subheader("Sanierungskosten")
            renovation_costs = st.number_input(
                "Sanierungskosten gesamt (EUR)",
                min_value=0.0,
                value=0.0,
                step=10_000.0,
                help="Kosten, die du als Sanierung/Modernisierung modellieren möchtest.",
            )
            renovation_eligible_pct = st.number_input(
                "Davon Sanierungs-AfA-fähig (%)",
                min_value=0.0,
                max_value=100.0,
                value=100.0,
                step=1.0,
                format="%.1f",
                help="Falls nur ein Teil nach Sonder-AfA läuft, den Rest ggf. als lineare AfA behandeln (hier vereinfacht: Rest wird zur Gebäudebasis addiert).",
            )
            renovation_afa_scheme = st.radio(
                "Sanierungs-AfA-Schema",
                options=["§7h/§7i (9%×8 + 7%×4)", "Benutzerdefiniert"],
                index=0,
                help="Klassisches Schema mit 100% über 12 Jahre (9% für 8 Jahre + 7% für 4 Jahre).",
            )
            if renovation_afa_scheme == "Benutzerdefiniert":
                renovation_custom = st.text_input(
                    "Jahres-% (kommagetrennt)",
                    value="9,9,9,9,9,9,9,9,7,7,7,7",
                    help="Beispiel: 9,9,9,9,9,9,9,9,7,7,7,7 (Summe 100%).",
                )
                perc_tmp = sanitize_percent_list(renovation_custom)
                perc_sum = sum(perc_tmp) * 100.0
                if perc_tmp:
                    if abs(perc_sum - 100.0) > 0.5:
                        st.warning(f"Dein Schema summiert sich auf {perc_sum:.1f}% (nicht 100%).")
                    else:
                        st.info(f"Schema-Summe: {perc_sum:.1f}%")

            acquisition_month = st.selectbox(
                "Anschaffungsmonat (AfA anteilig im Jahr 1)",
                options=list(range(1, 13)),
                index=0,
                help="Vereinfachung: AfA im Anschaffungsjahr wird monatsgenau anteilig gerechnet (Monat..Dezember).",
            )
            renovation_start_year = st.number_input(
                "Sanierungs-AfA Startjahr (1=sofort)",
                min_value=1,
                max_value=60,
                value=1,
                step=1,
                help="Vereinfachung: Sonder-AfA startet ab diesem Jahr (z.B. nach Fertigstellung).",
            )

            st.subheader("Laufende Kosten & Steuersatz")
            operating_costs_year = st.number_input(
                "Nicht-umlagefähige Kosten p.a. (EUR)",
                min_value=0.0,
                value=0.0,
                step=1_000.0,
                help="Vereinfachung: Verwaltung, Instandhaltung, Leerstand, etc. (sofern nicht umgelegt).",
            )
            tax_rate_de_pct = st.number_input(
                "Grenzsteuersatz (%)",
                min_value=0.0,
                max_value=100.0,
                value=42.0,
                step=1.0,
                format="%.1f",
            )
            income_cap_de_year = st.number_input(
                "Einkommen-Cap p.a. (EUR) (optional)",
                min_value=0.0,
                value=0.0,
                step=1_000.0,
                help="Optional: cappt Verlustverrechnung (sehr vereinfacht). 0 = kein Cap.",
            )

        years = int(
            st.number_input("Zeithorizont (Jahre)", min_value=1, max_value=60, value=30, step=1)
        )

        if tilgung_min_pct > tilgung_max_pct:
            st.warning("Tilgung min ist größer als Tilgung max – ich tausche intern die Werte.")

    # --- Core computations (minimal) ---
    purchase_price_f = float(st.session_state.purchase_price_num)
    if st.session_state.get("rent_mode", "total") == "sqm":
        rent_month_f = float(st.session_state.get("rent_total_calc", 0.0))
    else:
        rent_month_f = float(st.session_state.rent_num)
    interest_f = float(interest_pct) / 100.0
    t_min = float(min(tilgung_min_pct, tilgung_max_pct)) / 100.0
    t_max = float(max(tilgung_min_pct, tilgung_max_pct)) / 100.0
    loan_f = float(loan)
    living_area_f = float(living_area)
    commercial_area_f = float(commercial_area)
    total_area_f = float(total_area)
    rent_year = rent_month_f * 12.0

    payment_month_min = annuity_month(loan_f, interest_f, t_min)
    payment_month_max = annuity_month(loan_f, interest_f, t_max)

    gross_yield = safe_div(rent_year, purchase_price_f)
    grm = safe_div(purchase_price_f, rent_year)

    cashflow_month_min = rent_month_f - payment_month_min
    cashflow_month_max = rent_month_f - payment_month_max

    appreciation_rate = (float(appreciation_pct) / 100.0) if appreciation_enabled else 0.0

    # --- Tax / AfA computations ---
    tax_table: pd.DataFrame | None = None
    tax_table_min: pd.DataFrame | None = None
    tax_table_max: pd.DataFrame | None = None
    de_allocation: dict[str, float] | None = None
    depreciation_total_year: list[float] = [0.0 for _ in range(years)]

    # Simple AfA mode (keeps prior behavior): treat AfA as a positive cashflow shield (approx)
    simple_tax_shield_year = 0.0
    simple_afa_year = 0.0
    if ("tax_mode_ui" in st.session_state) and (st.session_state.tax_mode_ui == "Einfach"):
        if (
            afa_in_cashflow
            and building_share_pct > 0
            and afa_rate_pct > 0
            and tax_rate_pct > 0
        ):
            building_share = float(building_share_pct) / 100.0
            afa_rate = float(afa_rate_pct) / 100.0
            tax_rate = float(tax_rate_pct) / 100.0
            simple_afa_year = purchase_price_f * building_share * afa_rate
            income_cap = float(gross_income_year) if gross_income_year > 0 else 0.0
            taxable_base = min(simple_afa_year, income_cap) if income_cap > 0 else simple_afa_year
            simple_tax_shield_year = taxable_base * tax_rate
        depreciation_total_year = depreciation_linear(
            basis=purchase_price_f * (float(building_share_pct) / 100.0),
            rate=float(afa_rate_pct) / 100.0,
            years=years,
        )

    # Extended DE mode: allocation and schedules
    if ("tax_mode_ui" in st.session_state) and (st.session_state.tax_mode_ui == "Erweitert (DE)"):
        land_share = _clamp01(float(land_share_pct) / 100.0)
        total_acq = purchase_price_f * (1.0 + purchase_costs.total_rate())
        land_basis = total_acq * land_share
        building_basis = total_acq - land_basis

        ren_total = float(renovation_costs)
        ren_eligible = ren_total * _clamp01(float(renovation_eligible_pct) / 100.0)
        ren_rest = max(0.0, ren_total - ren_eligible)

        building_basis_total = max(0.0, building_basis + ren_rest)

        de_allocation = {
            "total_acq": float(total_acq),
            "purchase_costs": float(total_acq - purchase_price_f),
            "land_basis": float(land_basis),
            "building_basis": float(building_basis),
            "building_basis_total": float(building_basis_total),
            "renovation_total": float(ren_total),
            "renovation_eligible": float(ren_eligible),
            "renovation_rest": float(ren_rest),
        }

        building_dep = depreciation_linear(
            basis=building_basis_total,
            rate=float(building_afa_rate_pct) / 100.0,
            years=years,
        )

        if renovation_afa_scheme == "§7h/§7i (9%×8 + 7%×4)":
            perc = [0.09] * 8 + [0.07] * 4
        else:
            perc = sanitize_percent_list(renovation_custom)

        renovation_dep_raw = depreciation_schedule_percent(basis=ren_eligible, percents=perc, years=years)
        renovation_dep = shift_schedule(renovation_dep_raw, years=years, start_year=int(renovation_start_year))

        # Prorate year 1 AfA by acquisition month (simplified)
        building_dep = prorate_year1(building_dep, int(acquisition_month))
        renovation_dep = prorate_year1(renovation_dep, int(acquisition_month))

        depreciation_total_year = [float(building_dep[i] + renovation_dep[i]) for i in range(years)]

        # Build yearly interest/principal from real amortization for both tilgung scenarios
        sched_m_min = amortization_schedule_detailed(principal=loan_f, interest=interest_f, initial_repayment=t_min, years=years)
        sched_m_max = amortization_schedule_detailed(principal=loan_f, interest=interest_f, initial_repayment=t_max, years=years)
        y_min = aggregate_yearly_amortization(sched_m_min, years=years)
        y_max = aggregate_yearly_amortization(sched_m_max, years=years)

        def _build_tax_table(y_amort: pd.DataFrame) -> pd.DataFrame:
            rows: list[dict[str, float]] = []
            tax_rate = _clamp01(float(tax_rate_de_pct) / 100.0)
            cap = float(income_cap_de_year) if float(income_cap_de_year) > 0 else None

            for i in range(years):
                year_no = int(i + 1)
                interest_y = float(y_amort.loc[i, "interest"]) if i < len(y_amort) else 0.0
                principal_y = float(y_amort.loc[i, "principal"]) if i < len(y_amort) else 0.0
                dep = float(depreciation_total_year[i])

                before_tax = rent_year - interest_y - float(operating_costs_year)
                taxable = before_tax - dep
                effective_taxable = taxable
                if cap is not None:
                    if effective_taxable < 0:
                        effective_taxable = -min(abs(effective_taxable), cap)
                    else:
                        effective_taxable = min(effective_taxable, cap)
                tax = effective_taxable * tax_rate

                # Cashflow after financing: rent - operating - interest - principal - tax
                after_fin = rent_year - float(operating_costs_year) - interest_y - principal_y - tax

                rows.append(
                    {
                        "year": float(year_no),
                        "rent": float(rent_year),
                        "operating_costs": float(operating_costs_year),
                        "interest": interest_y,
                        "principal": principal_y,
                        "depreciation": dep,
                        "taxable": taxable,
                        "tax": tax,
                        "cashflow_after_tax_fin": after_fin,
                    }
                )

            return pd.DataFrame(rows)

        tax_table_min = _build_tax_table(y_min)
        tax_table_max = _build_tax_table(y_max)
        # Keep a single table for older UI blocks (first scenario)
        tax_table = tax_table_min

    # capital development schedules
    sched_min = amortization_schedule(
        principal=loan_f,
        interest=interest_f,
        initial_repayment=t_min,
        property_value=purchase_price_f,
        appreciation_rate=appreciation_rate,
        years=years,
    )
    sched_max = amortization_schedule(
        principal=loan_f,
        interest=interest_f,
        initial_repayment=t_max,
        property_value=purchase_price_f,
        appreciation_rate=appreciation_rate,
        years=years,
    )

    # --- Layout ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Kaufpreis", eur(purchase_price_f))
        st.metric("Darlehen", eur(loan_f))

    with k2:
        st.metric("Miete/Monat", eur(rent_month_f))
        st.metric("Bruttorendite p.a.", pct(gross_yield))

    with k3:
        st.metric("Faktor (Kaufpreis / Jahresmiete)", f"{grm:.2f}" if not math.isnan(grm) else "–")
        st.metric("Annuität/Monat (Tilgung min)", eur(payment_month_min))

    with k4:
        st.metric("Annuität/Monat (Tilgung max)", eur(payment_month_max))
        if ("tax_mode_ui" in st.session_state) and (st.session_state.tax_mode_ui == "Einfach") and afa_in_cashflow:
            cashflow_month_min_adj = cashflow_month_min + (simple_tax_shield_year / 12.0)
            cashflow_month_max_adj = cashflow_month_max + (simple_tax_shield_year / 12.0)
            st.metric("Cashflow/Monat inkl. Steuerwirkung (min .. max)", f"{eur(cashflow_month_min_adj)} .. {eur(cashflow_month_max_adj)}")
        elif ("tax_mode_ui" in st.session_state) and (st.session_state.tax_mode_ui == "Erweitert (DE)") and tax_table is not None:
            # Year-1 after-tax cashflow approximation
            # Here: cashflow_after_tax_fin already includes principal (i.e., after financing)
            cashflow_month_min_tax = float(tax_table_min.loc[0, "cashflow_after_tax_fin"]) / 12.0 if tax_table_min is not None else float("nan")
            cashflow_month_max_tax = float(tax_table_max.loc[0, "cashflow_after_tax_fin"]) / 12.0 if tax_table_max is not None else float("nan")
            st.metric(
                "Cashflow/Monat nach Steuer (min .. max)",
                f"{eur(cashflow_month_min_tax)} .. {eur(cashflow_month_max_tax)}",
            )
        else:
            st.metric("Cashflow/Monat (min .. max)", f"{eur(cashflow_month_min)} .. {eur(cashflow_month_max)}")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("€/m² (Kaufpreis / Wohnfläche)", eur(safe_div(purchase_price_f, living_area_f)))
    with a2:
        st.metric("€/m² (Kaufpreis / Gesamtfläche)", eur(safe_div(purchase_price_f, total_area_f)))

    if st.session_state.get("rent_mode", "total") == "sqm":
        rent_res_sqm = float(st.session_state.get("rent_res_sqm", 0.0))
        rent_com_sqm = float(st.session_state.get("rent_com_sqm", 0.0))
        rent_weighted = safe_div(rent_month_f, total_area_f)
        with a3:
            st.metric("Miete Wohnen (EUR/m²)", eur(rent_res_sqm))
        with a4:
            if commercial_area_f > 0:
                st.metric("Miete Gewerbe (EUR/m²)", eur(rent_com_sqm))
            else:
                st.metric("Miete gesamt (gewichtet) EUR/m²", eur(rent_weighted))
        if commercial_area_f > 0:
            st.caption(f"Gesamtmiete gewichtet: {eur(rent_weighted)} / m²")
    else:
        with a3:
            st.metric("Gesamtmiete €/m² (bez. auf Wohnfläche)", eur(safe_div(rent_month_f, living_area_f)))
        with a4:
            st.metric("Gesamtmiete €/m² (bez. auf Gesamtfläche)", eur(safe_div(rent_month_f, total_area_f)))

    if commercial_area_f > 0:
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Wohnfläche (m²)", f"{living_area_f:.0f}")
        with b2:
            st.metric("Gewerbefläche (m²)", f"{commercial_area_f:.0f}")
        with b3:
            st.metric("Gesamtfläche (m²)", f"{total_area_f:.0f}")
        with b4:
            st.metric("Gewerbe aktiv", "Ja")
    else:
        st.caption("Gewerbefläche ist deaktiviert (Toggle links).")

    st.caption(
        "Hinweis: Ohne separate Angabe der Wohn- und Gewerbemiete kann nicht sauber die reine Wohnmiete €/m² bzw. Gewerbemiete €/m² berechnet werden. "
        "Die Kennzahl oben teilt die *Gesamtkaltmiete* durch die jeweilige Fläche."
    )

    st.divider()

    if ("tax_mode_ui" in st.session_state) and (st.session_state.tax_mode_ui == "Erweitert (DE)") and (de_allocation is not None):
        st.subheader("Aufteilung: Grund · Altbausubstanz · Sanierung (DE, vereinfacht)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Grund & Boden (nicht AfA)", eur(de_allocation["land_basis"]))
        with c2:
            st.metric("Altbau/Gebäude-Basis", eur(de_allocation["building_basis"]))
        with c3:
            st.metric("Sanierung (Sonder-AfA Basis)", eur(de_allocation["renovation_eligible"]))
        with c4:
            st.metric("Sanierung (Rest → Gebäude)", eur(de_allocation["renovation_rest"]))
        st.caption(
            f"Kaufnebenkosten (modelliert): {eur(de_allocation['purchase_costs'])} | Anschaffung gesamt: {eur(de_allocation['total_acq'])}"
        )

    left, right = st.columns([1.05, 0.95])

    with left:
        st.subheader("Sweet-Spot-Diagramm: erforderliche Miete vs. Kaufpreis")
        st.caption("Band = Break-even-Miete (Miete = Annuität) bei deinem Zins und deiner Tilgung (min..max).")

        price_grid = pd.Series(range(100_000, 3_000_001, 25_000), name="price").astype(float)
        loan_ratio = safe_div(loan_f, purchase_price_f)
        loan_grid = (price_grid * loan_ratio).astype(float)
        req_rent_min = (loan_grid * (interest_f + t_min) / 12.0).astype(float)
        req_rent_max = (loan_grid * (interest_f + t_max) / 12.0).astype(float)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=price_grid,
                y=req_rent_min,
                mode="lines",
                name="Break-even (Tilgung min)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=price_grid,
                y=req_rent_max,
                mode="lines",
                name="Break-even (Tilgung max)",
                fill="tonexty",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[purchase_price_f],
                y=[rent_month_f],
                mode="markers",
                name="Dein Punkt (Kaufpreis, Miete)",
                marker=dict(size=12),
            )
        )
        fig.update_layout(
            xaxis_title="Kaufpreis (EUR)",
            yaxis_title="Kaltmiete (EUR/Monat)",
            hovermode="x",
        )
        st.plotly_chart(fig, width="stretch")

        # Quick interpretation
        if rent_month_f >= payment_month_max:
            st.success("Miete >= Annuität (auch bei Tilgung max): vereinfachtes Break-even erreicht.")
        elif rent_month_f >= payment_month_min:
            st.warning("Bei Tilgung min Break-even, bei Tilgung max noch darunter.")
        else:
            st.error("Miete liegt unter der Annuität (auch bei Tilgung min).")

    with right:
        st.subheader("Kapitalentwicklung (Eigenkapital durch Tilgung)")
        if appreciation_enabled:
            st.caption(
                "Property-Equity = Objektwert (inkl. Wertsteigerung) − Restschuld. Zusätzlich: kumulierter Cashflow aus Miete − Annuität (ohne Kosten)."
            )
        else:
            st.caption(
                "Property-Equity = Objektwert − Restschuld. Zusätzlich: kumulierter Cashflow aus Miete − Annuität (ohne Kosten)."
            )

        # Time series: equity + debt
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=sched_min["year"],
                y=sched_min["remaining_debt"],
                mode="lines",
                name="Restschuld (Tilgung min)",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=sched_max["year"],
                y=sched_max["remaining_debt"],
                mode="lines",
                name="Restschuld (Tilgung max)",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=sched_min["year"],
                y=sched_min["property_equity"],
                mode="lines",
                name="Property-Equity (Tilgung min)",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=sched_max["year"],
                y=sched_max["property_equity"],
                mode="lines",
                name="Property-Equity (Tilgung max)",
            )
        )
        if appreciation_enabled:
            fig2.add_trace(
                go.Scatter(
                    x=sched_min["year"],
                    y=sched_min["property_value"],
                    mode="lines",
                    name="Objektwert",
                    line=dict(dash="dot"),
                )
            )
        fig2.update_layout(xaxis_title="Jahre", yaxis_title="EUR")
        st.plotly_chart(fig2, width="stretch")

        # Cumulative cashflow lines (simple, constant rent/payment)
        cf_year_min = (rent_month_f - payment_month_min) * 12.0
        cf_year_max = (rent_month_f - payment_month_max) * 12.0
        years_axis = sched_min["year"].to_numpy()
        cum_cf_min = years_axis * cf_year_min
        cum_cf_max = years_axis * cf_year_max

        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=years_axis,
                y=cum_cf_min,
                mode="lines",
                name=f"Kum. Cashflow (Tilgung min {t_min*100:.2f}%)",
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=years_axis,
                y=cum_cf_max,
                mode="lines",
                name=f"Kum. Cashflow (Tilgung max {t_max*100:.2f}%)",
            )
        )

        # Simple mode: approximate tax shield as constant positive yearly add-on
        if ("tax_mode_ui" in st.session_state) and (st.session_state.tax_mode_ui == "Einfach") and afa_in_cashflow and simple_tax_shield_year > 0:
            cum_cf_min_tax = years_axis * (cf_year_min + simple_tax_shield_year)
            cum_cf_max_tax = years_axis * (cf_year_max + simple_tax_shield_year)

            fig3.add_trace(
                go.Scatter(
                    x=years_axis,
                    y=cum_cf_min_tax,
                    mode="lines",
                    name="Kum. Cashflow inkl. Steuerwirkung (Tilgung min)",
                    line=dict(dash="dot"),
                )
            )
            fig3.add_trace(
                go.Scatter(
                    x=years_axis,
                    y=cum_cf_max_tax,
                    mode="lines",
                    name="Kum. Cashflow inkl. Steuerwirkung (Tilgung max)",
                    line=dict(dash="dot"),
                )
            )
            st.caption(
                "Einfacher AfA-Modus (vereinfacht): "
                f"AfA {eur(simple_afa_year)} p.a. | Steuerwirkung {eur(simple_tax_shield_year)} p.a."
            )

        # Extended DE mode: year-by-year taxes with AfA split; plot cumulative after-tax cashflow (annual points)
        if ("tax_mode_ui" in st.session_state) and (st.session_state.tax_mode_ui == "Erweitert (DE)") and (tax_table_min is not None) and (tax_table_max is not None):
            y_idx = tax_table_min["year"].to_numpy()
            cum_min = tax_table_min["cashflow_after_tax_fin"].cumsum().to_numpy()
            cum_max = tax_table_max["cashflow_after_tax_fin"].cumsum().to_numpy()

            fig3.add_trace(
                go.Scatter(
                    x=y_idx,
                    y=cum_min,
                    mode="lines+markers",
                    name="Kum. Cashflow nach Steuer (Tilgung min)",
                    line=dict(dash="dot"),
                )
            )
            fig3.add_trace(
                go.Scatter(
                    x=y_idx,
                    y=cum_max,
                    mode="lines+markers",
                    name="Kum. Cashflow nach Steuer (Tilgung max)",
                    line=dict(dash="dot"),
                )
            )

        cashflow_title = "Kumulierter Cashflow (ohne Kosten)"
        if ("tax_mode_ui" in st.session_state) and st.session_state.tax_mode_ui in ("Einfach", "Erweitert (DE)"):
            cashflow_title = "Kumulierter Cashflow (Steuern optional)"
        fig3.update_layout(xaxis_title="Jahre", yaxis_title="EUR", title=cashflow_title)
        st.plotly_chart(fig3, width="stretch")

        if ("tax_mode_ui" in st.session_state) and (st.session_state.tax_mode_ui == "Erweitert (DE)"):
            st.subheader("Steuer-/AfA-Übersicht (vereinfachtes Modell)")
            st.caption(
                "Die Tabelle zeigt eine vereinfachte Steuerrechnung mit AfA-Aufteilung: Grund & Boden (0 AfA), Gebäude/Altbau (linear), Sanierung (Schema). "
                "Zinsen/Tilgung basieren auf dem Tilgungsplan (jährlich aggregiert)."
            )
            if (tax_table_min is not None) and (tax_table_max is not None):
                tab_min, tab_max = st.tabs(["Tilgung min", "Tilgung max"])
                show_cols = [
                    "year",
                    "rent",
                    "operating_costs",
                    "interest",
                    "principal",
                    "depreciation",
                    "taxable",
                    "tax",
                    "cashflow_after_tax_fin",
                ]
                with tab_min:
                    st.dataframe(tax_table_min[show_cols], width="stretch")
                with tab_max:
                    st.dataframe(tax_table_max[show_cols], width="stretch")



if __name__ == "__main__":
    main()
