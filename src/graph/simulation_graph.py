"""
DEPRECATED - not used at runtime.

The process topology lives in src/process/loan_application.py and is
executed by the DES engine (src/queue/simulation_engine.py) through
src/queue/step_executor.py.

Design note
    LangGraph was evaluated as the process orchestrator. Its
    run-to-completion execution model - `graph.invoke()` runs an entire
    case from START to END - is incompatible with discrete event
    simulation, where several cases progress concurrently while sharing
    a limited pool of agent resources. The DES engine therefore
    replicates LangGraph's routing semantics (conditional edges, state
    merge with append-only fields) and adds the temporal orchestration
    LangGraph does not provide: event queues, working hours, and case
    interleaving.
"""
