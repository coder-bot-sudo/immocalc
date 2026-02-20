"""Create matplotlib charts for the Hirschau property financing analysis.

Outputs:
- build/immobilien_charts.png
- build/immobilien_charts.pdf

The charts are formula-based from the known exposé facts:
- Purchase price: 1,190,000 EUR
- Current cold rent: 3,626 EUR/month
- Interest: 4% p.a.
- Tilgung range: 1% .. 1.5% p.a.
- Broker: 3.57% (rate)
- GrESt Bavaria: 3.5% (rate)
- Potential: 120 m²
- Unrented garages: 2

No assumptions about taxes, operating costs, vacancies, build-out costs, or
market rents are introduced. We only compute required *minimum* extra cold rent
needed to cover the annuity.

Run:
  /home/q621247/workspace/aws-log-uploader/.venv/bin/python tool/immobilie_plots.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Inputs:
    purchase_price_eur: float = 1_190_000.0
    rent_cold_month_eur: float = 3_626.0
    interest_rate: float = 0.04

    tilgung_min: float = 0.01
    tilgung_max: float = 0.015
    tilgung_step: float = 0.0005  # 0.05%-Punkte

    broker_rate: float = 0.0357
    grt_rate_bavaria: float = 0.035

    potential_sqm: float = 120.0
    garages_unrented: float = 2.0


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    x = start
    # inclusive upper bound
    while x <= stop + 1e-12:
        values.append(round(x, 6))
        x += step
    return values


def annuity_month(principal: float, interest: float, tilgung: float) -> float:
    return principal * (interest + tilgung) / 12.0


def main() -> None:
    p = Inputs()

    tilg = frange(p.tilgung_min, p.tilgung_max, p.tilgung_step)

    rent = [p.rent_cold_month_eur for _ in tilg]

    principal_a = p.purchase_price_eur  # only purchase price financed
    principal_b = p.purchase_price_eur * (1.0 + p.broker_rate + p.grt_rate_bavaria)  # incl. broker+GrESt

    rate_a = [annuity_month(principal_a, p.interest_rate, t) for t in tilg]
    rate_b = [annuity_month(principal_b, p.interest_rate, t) for t in tilg]

    gap_a = [ra - p.rent_cold_month_eur for ra in rate_a]
    gap_b = [rb - p.rent_cold_month_eur for rb in rate_b]

    # Requirements if only 120m² covers the gap
    req_eur_per_sqm_a = [g / p.potential_sqm for g in gap_a]
    req_eur_per_sqm_b = [g / p.potential_sqm for g in gap_b]

    # Requirements if only unrented garages cover the gap
    req_per_garage_a = [g / p.garages_unrented for g in gap_a]
    req_per_garage_b = [g / p.garages_unrented for g in gap_b]

    # Percent increase if only existing rent increases
    req_pct_a = [g / p.rent_cold_month_eur for g in gap_a]
    req_pct_b = [g / p.rent_cold_month_eur for g in gap_b]

    out_dir = Path("build")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tilg, rate_a, label="Annuität/Monat (A: Darlehen=Kaufpreis)")
    ax1.plot(tilg, rate_b, label="Annuität/Monat (B: inkl. Makler+GrESt)")
    ax1.plot(tilg, rent, label="Ist-Kaltmiete/Monat", linestyle="--")
    ax1.set_title("Annuität vs. Ist-Kaltmiete")
    ax1.set_xlabel("Tilgung p.a.")
    ax1.set_ylabel("EUR/Monat")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(tilg, gap_a, label="Lücke/Monat (A)")
    ax2.plot(tilg, gap_b, label="Lücke/Monat (B)")
    ax2.axhline(0.0, color="black", linewidth=1)
    ax2.set_title("Monatliche Deckungslücke (nur Annuität)")
    ax2.set_xlabel("Tilgung p.a.")
    ax2.set_ylabel("EUR/Monat")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(tilg, req_eur_per_sqm_a, label="€/m²/Monat nötig (A)")
    ax3.plot(tilg, req_eur_per_sqm_b, label="€/m²/Monat nötig (B)")
    ax3.set_title("Mindest-Zusatzmiete, wenn nur 120 m² Ausbau die Lücke decken")
    ax3.set_xlabel("Tilgung p.a.")
    ax3.set_ylabel("EUR/m²/Monat")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(tilg, req_per_garage_a, label="€/Garage/Monat nötig (A)")
    ax4.plot(tilg, req_per_garage_b, label="€/Garage/Monat nötig (B)")
    ax4b = ax4.twinx()
    ax4b.plot(tilg, [x * 100 for x in req_pct_a], label="% Mietanstieg nötig (A)", linestyle=":")
    ax4b.plot(tilg, [x * 100 for x in req_pct_b], label="% Mietanstieg nötig (B)", linestyle=":")

    ax4.set_title("Nur Garagen oder nur Bestandsmieten als Hebel")
    ax4.set_xlabel("Tilgung p.a.")
    ax4.set_ylabel("EUR/Garage/Monat")
    ax4b.set_ylabel("% der Ist-Kaltmiete")

    ax4.grid(True, alpha=0.3)

    # Combine legends of ax4 and ax4b
    handles1, labels1 = ax4.get_legend_handles_labels()
    handles2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle("Hirschau – Finanzierung 4% Zins, Tilgung 1%..1.5% (faktenbasiert)")

    png = out_dir / "immobilien_charts.png"
    pdf = out_dir / "immobilien_charts.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
