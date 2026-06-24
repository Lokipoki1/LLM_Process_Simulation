"""
simulation_controller.py
------------------------
Orquesta la ejecucion de N casos a traves del grafo LangGraph.
"""

from __future__ import annotations
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from .graph.simulation_graph import build_graph, make_initial_state
from .clock.simulation_clock import SimulationClock, extract_bpic_distributions
from .state import LoanCase


LOAN_GOALS = ["car", "home_improvement", "debt_consolidation", "education", "other"]


def generate_synthetic_case(case_id: str, rng: np.random.Generator) -> LoanCase:
    """Genera un caso sintetico con distribuciones aproximadas al BPIC 2012/2017."""
    monthly_income   = float(rng.lognormal(mean=8.1, sigma=0.4))
    amount_requested = float(rng.lognormal(mean=9.8, sigma=0.7))
    number_of_terms  = int(rng.choice([12, 24, 36, 48, 60, 72]))
    monthly_cost     = round(amount_requested / number_of_terms * 1.06, 2)
    credit_score     = int(np.clip(rng.normal(loc=660, scale=80), 300, 850))
    loan_goal        = str(rng.choice(LOAN_GOALS))
    return LoanCase(
        case_id=case_id,
        amount_requested=round(amount_requested, 2),
        loan_goal=loan_goal,
        number_of_terms=number_of_terms,
        monthly_cost=monthly_cost,
        credit_score=credit_score,
        monthly_income=round(monthly_income, 2),
        applicant_id=f"APP-{case_id}",
    )


def load_bpic_cases(xes_path: str, max_cases: int | None = None) -> list[LoanCase]:
    """Carga casos reales del BPIC 2012/2017 desde un archivo XES."""
    import pm4py
    log    = pm4py.read_xes(xes_path)
    df     = pm4py.convert_to_dataframe(log)
    cases_df = df.groupby("case:concept:name").first().reset_index()
    if max_cases:
        cases_df = cases_df.head(max_cases)
    rng = np.random.default_rng(seed=42)
    cases = []
    for _, row in cases_df.iterrows():
        case_id = str(row["case:concept:name"])
        amount  = float(row.get("AMOUNT_REQ", row.get("RequestedAmount", rng.lognormal(9.8, 0.7))))
        terms   = int(row.get("NumberOfTerms", rng.choice([24, 36, 48])))
        cases.append(LoanCase(
            case_id=case_id,
            amount_requested=round(amount, 2),
            loan_goal=str(row.get("LoanGoal", rng.choice(LOAN_GOALS))),
            number_of_terms=terms,
            monthly_cost=round(amount / terms * 1.06, 2),
            credit_score=int(np.clip(rng.normal(660, 80), 300, 850)),
            monthly_income=float(rng.lognormal(8.1, 0.4)),
            applicant_id=f"APP-{case_id}",
        ))
    return cases


@dataclass
class CaseResult:
    case_id:        str
    status:         str
    revision_count: int
    n_actions:      int
    n_log_entries:  int
    duration_s:     float
    event_log:      list = field(default_factory=list)


class SimulationController:
    """
    Corre N casos y acumula el event log global.

    Uso:
        ctrl = SimulationController(n_cases=50)
        ctrl.run()
        ctrl.export_xes("output/simulation.xes")
    """

    def __init__(
        self,
        n_cases:         int       = 50,
        model:           str       = "llama3.1",
        ollama_base_url: str       = "http://localhost:11434",
        bpic_xes_path:   str|None  = None,
        sim_start_ts:    str       = "2012-01-02T08:00:00",
        seed:            int       = 42,
        output_dir:      str       = "output",
    ):
        self.n_cases         = n_cases
        self.model           = model
        self.ollama_base_url = ollama_base_url
        self.bpic_xes_path   = bpic_xes_path
        self.sim_start_ts    = sim_start_ts
        self.seed            = seed
        self.output_dir      = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._results:   list[CaseResult] = []
        self._event_log: list[dict]       = []

    def _prepare_cases(self) -> list[LoanCase]:
        if self.bpic_xes_path and Path(self.bpic_xes_path).exists():
            print(f"  Cargando casos BPIC desde: {self.bpic_xes_path}")
            return load_bpic_cases(self.bpic_xes_path, max_cases=self.n_cases)
        print(f"  Generando {self.n_cases} casos sinteticos (seed={self.seed})")
        rng = np.random.default_rng(self.seed)
        return [generate_synthetic_case(f"LOAN-{i:04d}", rng) for i in range(1, self.n_cases + 1)]

    def _prepare_distributions(self) -> dict | None:
        if self.bpic_xes_path and Path(self.bpic_xes_path).exists():
            print("  Extrayendo distribuciones del BPIC...")
            return extract_bpic_distributions(self.bpic_xes_path)
        return None

    def run(self) -> list[CaseResult]:
        print(f"\n{'='*55}")
        print(f"  Simulacion Multi-LLM-Agent BPS")
        print(f"  Modelo: {self.model} | Casos: {self.n_cases}")
        print(f"{'='*55}\n")

        cases  = self._prepare_cases()
        dists  = self._prepare_distributions()
        clock  = SimulationClock(start_timestamp=self.sim_start_ts, distributions=dists, seed=self.seed)
        graph  = build_graph(model=self.model, ollama_base_url=self.ollama_base_url, clock=clock)

        approved = rejected = errors = 0

        for i, case in enumerate(cases, 1):
            print(f"  [{i:3d}/{self.n_cases}] {case['case_id']} | "
                  f"EUR {case['amount_requested']:>8,.0f} | score {case['credit_score']}", end=" ", flush=True)
            t0 = time.time()
            try:
                final = graph.invoke(make_initial_state(case), config={"recursion_limit": 40})
                dur   = time.time() - t0
                status = final["status"]
                result = CaseResult(
                    case_id=case["case_id"], status=status,
                    revision_count=final["revision_count"],
                    n_actions=len(final["agent_history"]),
                    n_log_entries=len(final["event_log"]),
                    duration_s=dur, event_log=final["event_log"],
                )
                self._results.append(result)
                self._event_log.extend(final["event_log"])
                if status == "approved":   approved += 1
                elif status == "rejected": rejected += 1
                print(f"-> {status.upper():8s} | {result.n_actions} acciones | {dur:.1f}s")
            except Exception as e:
                errors += 1
                print(f"-> ERROR: {e}")
            if i % 10 == 0:
                self._save_partial(i)

        print(f"\n{'─'*55}")
        print(f"  Completados: {len(self._results)} | Aprobados: {approved} | Rechazados: {rejected} | Errores: {errors}")
        print(f"  Entradas event log: {len(self._event_log)}")
        print(f"{'='*55}\n")
        return self._results

    def export_xes(self, filename: str = "simulation.xes") -> Path:
        """Exporta el event log a formato XES via PM4Py."""
        import pm4py
        if not self._event_log:
            raise RuntimeError("Sin datos — ejecuta run() primero.")
        df = pd.DataFrame(self._event_log).rename(columns={
            "case_concept_name":    "case:concept:name",
            "concept_name":         "concept:name",
            "time_timestamp":       "time:timestamp",
            "org_resource":         "org:resource",
            "lifecycle_transition": "lifecycle:transition",
        })
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
        out = self.output_dir / filename
        pm4py.write_xes(pm4py.convert_to_event_log(df), str(out))
        print(f"  XES exportado -> {out}")
        return out

    def export_json(self, filename: str = "simulation.json") -> Path:
        """Guarda el event log como JSON para debugging."""
        out = self.output_dir / filename
        with open(out, "w") as f:
            json.dump(self._event_log, f, indent=2, default=str)
        print(f"  JSON exportado -> {out}")
        return out

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "case_id": r.case_id, "status": r.status,
            "revision_count": r.revision_count, "n_actions": r.n_actions,
            "n_log_entries": r.n_log_entries, "duration_s": round(r.duration_s, 2),
        } for r in self._results])

    def _save_partial(self, up_to: int):
        with open(self.output_dir / f"partial_{up_to:04d}.json", "w") as f:
            json.dump(self._event_log, f, default=str)
