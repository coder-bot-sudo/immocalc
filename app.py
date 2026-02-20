"""Interactive dashboard: sweet spot between purchase price and rent.

Goal
- Find the sweet spot between purchase price and monthly rent.
- Show capital development via amortization (and property equity).

Scope
- No "potential levers" or market assumptions.
- Parameters are editable as values (no sliders required).

Notes
- This is a simplified model (no operating costs, vacancy, taxes).

Run
    streamlit run app.py
"""

from __future__ import annotations

import math
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
    years: int,
) -> pd.DataFrame:
    payment = annuity_month(principal, interest, initial_repayment)
    months = years * 12

    remaining = principal
    rows: list[dict[str, float]] = []

    for m in range(months + 1):
        year = m / 12.0
        amortized_equity = principal - remaining
        property_equity = property_value - remaining
        rows.append(
            {
                "year": year,
                "remaining_debt": remaining,
                "amortized_equity": amortized_equity,
                "property_equity": property_equity,
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
                rows.append(
                    {
                        "year": year2,
                        "remaining_debt": 0.0,
                        "amortized_equity": principal,
                        "property_equity": property_value,
                        "payment": payment,
                    }
                )
            break

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Immobilien-Kennzahlen", layout="wide")

    DEFAULT_PRICE = 1_190_000.0
    DEFAULT_RENT = 3_626.0
    DEFAULT_INTEREST = 0.04
    DEFAULT_TILGUNG_MIN = 0.01
    DEFAULT_TILGUNG_MAX = 0.015
    DEFAULT_LIVING_AREA = 370.0
    DEFAULT_TOTAL_AREA = 520.0

    st.title("Sweet Spot: Kaufpreis ↔ Miete (Kapitalentwicklung durch Tilgung)")
    st.caption(
        "Vereinfachtes Modell ohne Kosten/Steuern: du kannst Kaufpreis, Miete, Zins, Tilgung (Min/Max), Flächen und optional das Darlehen direkt als Werte ändern."
    )

    with st.sidebar:
        st.header("Inputs")
        purchase_price = st.slider(
            "Kaufpreis (EUR)",
            min_value=100_000,
            max_value=3_000_000,
            value=int(DEFAULT_PRICE),
            step=10_000,
        )
        rent_cold_month = st.slider(
            "Kaltmiete (EUR/Monat)",
            min_value=0,
            max_value=12_000,
            value=int(DEFAULT_RENT),
            step=50,
        )

        st.subheader("Finanzierung")
        interest = st.number_input("Sollzins p.a.", min_value=0.0, max_value=0.25, value=DEFAULT_INTEREST, step=0.0005, format="%.4f")
        tilgung_min = st.number_input(
            "Tilgung min p.a.", min_value=0.0, max_value=0.25, value=DEFAULT_TILGUNG_MIN, step=0.0005, format="%.4f"
        )
        tilgung_max = st.number_input(
            "Tilgung max p.a.", min_value=0.0, max_value=0.25, value=DEFAULT_TILGUNG_MAX, step=0.0005, format="%.4f"
        )
        loan = st.number_input(
            "Darlehen (EUR)",
            min_value=0.0,
            value=float(purchase_price),
            step=10_000.0,
            help="Default = Kaufpreis. Wenn kleiner, entspricht das Eigenkapital im Objekt (vereinfachend).",
        )

        st.subheader("Größe")
        living_area = st.number_input("Wohnfläche (m²)", min_value=0.0, value=DEFAULT_LIVING_AREA, step=5.0)
        total_area = st.number_input("Gesamtfläche (m²)", min_value=0.0, value=DEFAULT_TOTAL_AREA, step=5.0)

        years = int(
            st.number_input("Zeithorizont (Jahre)", min_value=1, max_value=60, value=30, step=1)
        )

        if tilgung_min > tilgung_max:
            st.warning("Tilgung min ist größer als Tilgung max – ich tausche intern die Werte.")

    # --- Core computations (minimal) ---
    purchase_price_f = float(purchase_price)
    rent_month_f = float(rent_cold_month)
    interest_f = float(interest)
    t_min = float(min(tilgung_min, tilgung_max))
    t_max = float(max(tilgung_min, tilgung_max))
    loan_f = float(loan)
    living_area_f = float(living_area)
    total_area_f = float(total_area)
    rent_year = rent_month_f * 12.0

    payment_month_min = annuity_month(loan_f, interest_f, t_min)
    payment_month_max = annuity_month(loan_f, interest_f, t_max)

    gross_yield = safe_div(rent_year, purchase_price_f)
    grm = safe_div(purchase_price_f, rent_year)

    cashflow_month_min = rent_month_f - payment_month_min
    cashflow_month_max = rent_month_f - payment_month_max

    # capital development schedules
    sched_min = amortization_schedule(
        principal=loan_f,
        interest=interest_f,
        initial_repayment=t_min,
        property_value=purchase_price_f,
        years=years,
    )
    sched_max = amortization_schedule(
        principal=loan_f,
        interest=interest_f,
        initial_repayment=t_max,
        property_value=purchase_price_f,
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
        st.metric("Cashflow/Monat (min .. max)", f"{eur(cashflow_month_min)} .. {eur(cashflow_month_max)}")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("€/m² (Kaufpreis / Wohnfläche)", eur(safe_div(purchase_price_f, living_area_f)))
    with a2:
        st.metric("€/m² (Kaufpreis / Gesamtfläche)", eur(safe_div(purchase_price_f, total_area_f)))
    with a3:
        st.metric("Miete €/m² (Wohnfläche)", eur(safe_div(rent_month_f, living_area_f)))
    with a4:
        st.metric("Miete €/m² (Gesamtfläche)", eur(safe_div(rent_month_f, total_area_f)))

    st.divider()

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
        st.caption(
            "Property-Equity = Kaufpreis − Restschuld (Objektwert konstant). Zusätzlich: kumulierter Cashflow aus Miete − Annuität (ohne Kosten)."
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
        fig2.update_layout(xaxis_title="Jahre", yaxis_title="EUR")
        st.plotly_chart(fig2, width="stretch")

        # Cumulative cashflow lines (simple, constant rent/payment)
        cf_year_min = (rent_month_f - payment_month_min) * 12.0
        cf_year_max = (rent_month_f - payment_month_max) * 12.0
        years_axis = sched_min["year"].to_numpy()
        cum_cf_min = years_axis * cf_year_min
        cum_cf_max = years_axis * cf_year_max

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=years_axis, y=cum_cf_min, mode="lines", name="Kum. Cashflow (Tilgung 1,0%)"))
        fig3.add_trace(go.Scatter(x=years_axis, y=cum_cf_max, mode="lines", name="Kum. Cashflow (Tilgung 1,5%)"))
        fig3.update_layout(xaxis_title="Jahre", yaxis_title="EUR", title="Kumulierter Cashflow (ohne Kosten)")
        st.plotly_chart(fig3, width="stretch")


if __name__ == "__main__":
    main()
