"""
validate_setup.py
-----------------
Corre esto para verificar que el stack base funciona antes de
instalar las dependencias pesadas (LangGraph, PM4Py).

No necesita API keys ni conexión a internet.
"""

import sys

def test_state():
    from src.state import ProcessState, LoanCase, AgentAction, XESEntry
    print("  ✓ ProcessState importado correctamente")

    # Crear un caso de prueba
    case: LoanCase = {
        "case_id": "TEST-001",
        "amount_requested": 15000.0,
        "loan_goal": "car",
        "number_of_terms": 36,
        "monthly_cost": 450.0,
        "credit_score": 680,
        "monthly_income": 3200.0,
        "applicant_id": "APP-001",
    }

    state: ProcessState = {
        "case": case,
        "status": "pending",
        "current_agent": "junior_clerk",
        "messages": [],
        "agent_history": [],
        "sim_clock": 0.0,
        "event_log": [],
        "revision_count": 0,
        "rejection_reason": None,
        "next_agent": None,
    }
    assert state["case"]["case_id"] == "TEST-001"
    print("  ✓ LoanCase y ProcessState instanciados correctamente")


def test_clock():
    from src.clock.simulation_clock import SimulationClock
    clock = SimulationClock(start_timestamp="2012-01-01T08:00:00", seed=42)
    duration, ts = clock.advance_for_activity("validate_application")
    assert duration > 0
    assert ts > clock.current - duration  # timestamp avanzó
    iso = clock.current_iso()
    assert "2012" in iso
    print(f"  ✓ SimulationClock OK — validate_application duró {duration/60:.1f} min")


def test_schemas():
    from src.tools.schemas import (
        IntakeApplication, CheckDocuments, ForwardCase,
        ValidateApplication, EscalateCase,
        AssessRisk, ApproveLoan, RejectLoan,
        AGENT_TOOLS,
    )
    # Validar que Pydantic acepta inputs correctos
    tool = IntakeApplication(
        case_id="TEST-001",
        applicant_acknowledged=True,
        initial_notes="Documentación aparentemente completa.",
    )
    assert tool.case_id == "TEST-001"
    assert len(AGENT_TOOLS["junior_clerk"]) == 3
    assert len(AGENT_TOOLS["senior_clerk"]) == 3
    assert len(AGENT_TOOLS["credit_officer"]) == 3
    print("  ✓ Pydantic schemas válidos y AGENT_TOOLS configurado correctamente")


def test_observer():
    from src.observer.observer import build_xes_entry
    from src.state import AgentAction
    action: AgentAction = {
        "agent_name": "junior_clerk",
        "tool_name": "IntakeApplication",
        "tool_input": {"case_id": "TEST-001"},
        "tool_output": {},
        "sim_timestamp": 1325404800.0,  # 2012-01-01T08:00:00 UTC
        "real_duration": 1800.0,
    }
    entry = build_xes_entry(action)
    assert entry["concept_name"] == "A_INTAKE"
    assert entry["org_resource"] == "Junior Clerk"
    assert entry["lifecycle_transition"] == "complete"
    assert "case_id" not in entry  # debe ser case_concept_name
    print("  ✓ ObserverModule genera XESEntry con campos PM4Py correctos")


if __name__ == "__main__":
    print("\n🔍 Validando stack base del framework...\n")
    tests = [test_state, test_clock, test_schemas, test_observer]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__} FALLÓ: {e}")
            failed += 1
    if failed == 0:
        print("\n✅ Todo OK — stack base listo para integrar LangGraph y LLM agents\n")
    else:
        print(f"\n❌ {failed} test(s) fallaron\n")
        sys.exit(1)
