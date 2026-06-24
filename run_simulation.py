"""
run_simulation.py
-----------------
Script de prueba: corre UN caso de préstamo a través del grafo.

Uso:
    python run_simulation.py

Asegúrate de que Ollama esté corriendo:
    ollama serve
    ollama pull llama3.1
"""

import os
import json
from dotenv import load_dotenv
from src.graph.simulation_graph import build_graph, make_initial_state
from src.clock.simulation_clock import SimulationClock

load_dotenv()

# ── Caso de prueba (basado en perfil BPIC 2012) ──────────────
TEST_CASE = {
    "case_id":           "LOAN-TEST-001",
    "amount_requested":  15000.0,
    "loan_goal":         "car",
    "number_of_terms":   36,
    "monthly_cost":      450.0,
    "credit_score":      680,
    "monthly_income":    3200.0,
    "applicant_id":      "APP-TEST-001",
}

def main():
    model   = os.getenv("LLM_MODEL", "llama3.1")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    print(f"\n{'='*55}")
    print(f"  Multi-LLM-Agent BPS — Simulación de prueba")
    print(f"  Modelo: {model} | URL: {base_url}")
    print(f"{'='*55}")
    print(f"\n  Caso: {TEST_CASE['case_id']}")
    print(f"  Monto: EUR {TEST_CASE['amount_requested']:,.0f}")
    print(f"  Score: {TEST_CASE['credit_score']}")
    print(f"  Ratio deuda/ingreso: {TEST_CASE['monthly_cost']/TEST_CASE['monthly_income']:.2f}")
    print(f"\n{'─'*55}\n")

    clock = SimulationClock(start_timestamp="2012-01-02T08:00:00", seed=42)
    graph = build_graph(model=model, ollama_base_url=base_url, clock=clock)

    initial_state = make_initial_state(TEST_CASE)

    # Correr la simulación
    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 25},
    )

    # ── Resultados ────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  RESULTADO: {final_state['status'].upper()}")
    print(f"  Revisiones: {final_state['revision_count']}")
    print(f"  Acciones ejecutadas: {len(final_state['agent_history'])}")
    print(f"  Entradas en event log: {len(final_state['event_log'])}")
    print(f"{'─'*55}")

    print("\n  Trazas del event log:")
    for entry in final_state["event_log"]:
        print(f"    [{entry['time_timestamp'][:19]}] "
              f"{entry['org_resource']:15s} → {entry['concept_name']}")

    print(f"\n{'='*55}\n")

    # Guardar event log como JSON (para inspección)
    with open("output_event_log.json", "w") as f:
        json.dump(final_state["event_log"], f, indent=2)
    print("  Event log guardado en: output_event_log.json\n")

if __name__ == "__main__":
    main()
