"""Interactive dashboard: sweet spot between purchase price and rent.

User intent (latest):
- Show how "capital" (equity in the property via amortization) develops.
- Remove all potential levers.
- Only two sliders: purchase price and monthly rent.
- A chart that updates while sliding, to find the sweet spot.

Assumptions kept explicit and minimal:
- Interest fixed at 4% p.a.
- Initial repayment range fixed at 1.0% .. 1.5% p.a. (band).
- Loan principal == purchase price (transaction costs ignored here).
- Property value assumed constant == purchase price (no appreciation).

Run:
    /home/q621247/workspace/aws-log-uploader/.venv/bin/streamlit run tool/immobilie_dashboard.py
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
    years: int,
) -> pd.DataFrame:
    payment = annuity_month(principal, interest, initial_repayment)
    months = years * 12

    remaining = principal
    rows: list[dict[str, float]] = []

    for m in range(months + 1):
        year = m / 12.0
        equity = principal - remaining
        rows.append({"year": year, "remaining_debt": remaining, "equity": equity, "payment": payment})

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
                rows.append({"year": year2, "remaining_debt": 0.0, "equity": principal, "payment": payment})
            break

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Immobilien-Kennzahlen", layout="wide")

    DEFAULT_PRICE = 1_190_000.0
    DEFAULT_RENT = 3_626.0
    INTEREST = 0.04
    TILGUNG_MIN = 0.01
    TILGUNG_MAX = 0.015

    st.title("Sweet Spot: Kaufpreis ↔ Miete (Kapitalentwicklung durch Tilgung)")
    st.caption(
        "Nur zwei Slider. Annahmen: 4% Zins p.a., Tilgung 1,0%–1,5% p.a., Darlehen = Kaufpreis, Objektwert konstant = Kaufpreis (keine Wertsteigerung)."
    )

    with st.sidebar:
        st.header("Parameter")
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

        st.caption("Fix: Zins 4% p.a. | Tilgung 1,0%–1,5% p.a. | Darlehen=Kaufpreis")

        years = st.slider("Zeithorizont (Jahre)", min_value=5, max_value=40, value=30, step=1)

    # --- Core computations (minimal) ---
    purchase_price_f = float(purchase_price)
    rent_month_f = float(rent_cold_month)
    rent_year = rent_month_f * 12.0

    loan = purchase_price_f
    payment_month_min = annuity_month(loan, INTEREST, TILGUNG_MIN)
    payment_month_max = annuity_month(loan, INTEREST, TILGUNG_MAX)

    gross_yield = safe_div(rent_year, purchase_price_f)
    grm = safe_div(purchase_price_f, rent_year)

    cashflow_month_min = rent_month_f - payment_month_min
    cashflow_month_max = rent_month_f - payment_month_max

    # capital development schedules
    sched_min = amortization_schedule(principal=loan, interest=INTEREST, initial_repayment=TILGUNG_MIN, years=years)
    sched_max = amortization_schedule(principal=loan, interest=INTEREST, initial_repayment=TILGUNG_MAX, years=years)

    # --- Layout ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Kaufpreis", eur(purchase_price_f))
        st.metric("Darlehen (vereinfachend)", eur(loan))

    with k2:
        st.metric("Miete/Monat", eur(rent_month_f))
        st.metric("Bruttorendite p.a.", pct(gross_yield))

    with k3:
        st.metric("Faktor (Kaufpreis / Jahresmiete)", f"{grm:.2f}" if not math.isnan(grm) else "–")
        st.metric("Annuität/Monat (Tilgung 1,0%)", eur(payment_month_min))

    with k4:
        st.metric("Annuität/Monat (Tilgung 1,5%)", eur(payment_month_max))
        st.metric("Cashflow/Monat (1,0% .. 1,5%)", f"{eur(cashflow_month_min)} .. {eur(cashflow_month_max)}")

    st.divider()

    left, right = st.columns([1.05, 0.95])

    with left:
        st.subheader("Sweet-Spot-Diagramm: erforderliche Miete vs. Kaufpreis")
        st.caption("Band = Break-even-Miete (Miete = Annuität) bei 4% Zins und 1,0%–1,5% Tilgung.")

        price_grid = pd.Series(range(100_000, 3_000_001, 25_000), name="price").astype(float)
        req_rent_min = (price_grid * (INTEREST + TILGUNG_MIN) / 12.0).astype(float)
        req_rent_max = (price_grid * (INTEREST + TILGUNG_MAX) / 12.0).astype(float)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=price_grid,
                y=req_rent_min,
                mode="lines",
                name="Break-even (Tilgung 1,0%)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=price_grid,
                y=req_rent_max,
                mode="lines",
                name="Break-even (Tilgung 1,5%)",
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
            st.success("Bei 1,5% Tilgung ist die Miete >= Annuität (vereinfachtes Break-even erreicht).")
        elif rent_month_f >= payment_month_min:
            st.warning("Bei 1,0% Tilgung Break-even, bei 1,5% noch darunter.")
        else:
            st.error("Miete liegt unter der Annuität (auch bei 1,0% Tilgung).")

    with right:
        st.subheader("Kapitalentwicklung (Eigenkapital durch Tilgung)")
        st.caption(
            "Eigenkapital = Kaufpreis − Restschuld (Objektwert konstant). Zusätzlich: kumulierter Cashflow aus Miete − Annuität (ohne Kosten)."
        )

        # Time series: equity + debt
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=sched_min["year"],
                y=sched_min["remaining_debt"],
                mode="lines",
                name="Restschuld (Tilgung 1,0%)",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=sched_max["year"],
                y=sched_max["remaining_debt"],
                mode="lines",
                name="Restschuld (Tilgung 1,5%)",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=sched_min["year"],
                y=sched_min["equity"],
                mode="lines",
                name="Eigenkapital (Tilgung 1,0%)",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=sched_max["year"],
                y=sched_max["equity"],
                mode="lines",
                name="Eigenkapital (Tilgung 1,5%)",
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
