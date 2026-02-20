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

        afa_enabled = st.toggle("Steuerliche Abschreibung (AfA) aktivieren", value=False)
        building_share_pct = st.number_input(
            "Gebäudeanteil am Kaufpreis (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            format="%.1f",
            disabled=not afa_enabled,
            help="Vereinfachung: AfA-Basis = Kaufpreis × Gebäudeanteil.",
        )
        afa_rate_pct = st.number_input(
            "AfA-Satz p.a. (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            disabled=not afa_enabled,
        )
        tax_rate_pct = st.number_input(
            "Grenzsteuersatz (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            format="%.1f",
            disabled=not afa_enabled,
            help="Vereinfachung: Steuerwirkung = AfA × Grenzsteuersatz.",
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
        st.metric("Cashflow/Monat (min .. max)", f"{eur(cashflow_month_min)} .. {eur(cashflow_month_max)}")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("€/m² (Kaufpreis / Wohnfläche)", eur(safe_div(purchase_price_f, living_area_f)))
    with a2:
        st.metric("€/m² (Kaufpreis / Gesamtfläche)", eur(safe_div(purchase_price_f, total_area_f)))
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

        if afa_enabled and building_share_pct > 0 and afa_rate_pct > 0 and tax_rate_pct > 0:
            building_share = float(building_share_pct) / 100.0
            afa_rate = float(afa_rate_pct) / 100.0
            tax_rate = float(tax_rate_pct) / 100.0
            afa_year = purchase_price_f * building_share * afa_rate
            tax_shield_year = afa_year * tax_rate

            cum_cf_min_tax = years_axis * (cf_year_min + tax_shield_year)
            cum_cf_max_tax = years_axis * (cf_year_max + tax_shield_year)

            fig3.add_trace(
                go.Scatter(
                    x=years_axis,
                    y=cum_cf_min_tax,
                    mode="lines",
                    name=f"Kum. Cashflow inkl. AfA (Tilgung min)",
                    line=dict(dash="dot"),
                )
            )
            fig3.add_trace(
                go.Scatter(
                    x=years_axis,
                    y=cum_cf_max_tax,
                    mode="lines",
                    name=f"Kum. Cashflow inkl. AfA (Tilgung max)",
                    line=dict(dash="dot"),
                )
            )
            st.caption(
                f"AfA (vereinfacht): {eur(afa_year)} p.a. | Steuerwirkung (AfA×Grenzsteuersatz): {eur(tax_shield_year)} p.a."
            )

        cashflow_title = "Kumulierter Cashflow (ohne Kosten)"
        if afa_enabled:
            cashflow_title = "Kumulierter Cashflow (ohne Kosten; AfA optional)"
        fig3.update_layout(xaxis_title="Jahre", yaxis_title="EUR", title=cashflow_title)
        st.plotly_chart(fig3, width="stretch")


if __name__ == "__main__":
    main()
