"""
DEPRECATED — not used at runtime. Safe to delete.

The process topology now lives in src/queue/process_graph.py and is
executed by the DES engine (src/queue/simulation_engine.py).

This file still referenced the pre-BPIC-2017 schema (`case`,
`revision_count`) and would raise on import against the current
state.py. It is kept only as a marker so a future refactor does not
mistake it for live code.

Design note for the thesis:
    LangGraph was evaluated as the process orchestrator. Its
    run-to-completion execution model — `graph.invoke()` runs an entire
    case from START to END — is incompatible with discrete event
    simulation, where several cases progress concurrently while sharing
    a limited pool of agent resources. The DES engine therefore
    replicates LangGraph's routing semantics (conditional edges, state
    merge with append-only fields) and adds the temporal orchestration
    LangGraph does not provide: event queues, working hours, and case
    interleaving. The LangGraph definition is retained in
    process_graph.py as the formal, renderable specification of the
    process topology.
"""
