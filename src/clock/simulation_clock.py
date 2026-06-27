"""
simulation_clock.py
--------------------
Reloj lógico de simulación.

Responsabilidad única: dado el nombre de una actividad, devuelve
una duración en segundos samplea de la distribución log-normal
estimada a partir del BPIC 2012/2017.

Esto es CRÍTICO para que las métricas AED y CTD sean válidas:
el LLM decide QUÉ actividad ejecutar, el reloj decide CUÁNTO tarda.
"""

from __future__ import annotations
import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────
# Parámetros log-normal por actividad
# Fuente: distribuciones aproximadas del BPIC 2012 (en segundos).
# Estos valores se refinarán en la fase de dataset analysis (días 15-28)
# usando extract_bpic_distributions() más abajo.
#
# Formato: { activity_name: (mu, sigma) } de la distribución log-normal
# donde mu y sigma son los parámetros de la distribución subyacente normal.
# ─────────────────────────────────────────────────────────────────

DEFAULT_DISTRIBUTIONS: dict[str, tuple[float, float]] = {
    # Junior Clerk — keyed by tool class name (PascalCase)
    "IntakeApplication":      (7.5, 0.8),   # ~1800s ≈ 30 min
    "CheckDocuments":         (7.8, 0.7),   # ~2400s ≈ 40 min
    "ForwardCase":            (6.2, 0.5),   # ~490s  ≈ 8 min
    "ReturnApplicationEarly": (6.0, 0.4),   # ~400s  ≈ 7 min

    # Senior Clerk
    "ValidateApplication":    (8.5, 0.9),   # ~4900s ≈ 80 min
    "RequestAdditionalInfo":  (9.2, 1.0),   # ~9900s ≈ 165 min (espera respuesta)
    "EscalateCase":           (6.5, 0.5),   # ~665s  ≈ 11 min

    # Credit Officer
    "AssessRisk":             (9.0, 0.8),   # ~8100s ≈ 135 min
    "ApproveLoan":            (7.2, 0.6),   # ~1300s ≈ 22 min
    "RejectLoan":             (7.0, 0.6),   # ~1100s ≈ 18 min

    # Fallback para actividades no reconocidas
    "_default":               (7.5, 1.0),
}


class SimulationClock:
    """
    Gestiona el tiempo lógico de la simulación.

    Uso:
        clock = SimulationClock(start_timestamp="2012-01-01T08:00:00")
        duration = clock.sample_duration("validate_application")
        clock.advance(duration)
        iso_ts = clock.current_iso()
    """

    def __init__(
        self,
        start_timestamp: str = "2012-01-01T08:00:00",
        distributions: dict[str, tuple[float, float]] | None = None,
        seed: int | None = None,
    ):
        from datetime import datetime
        self._current: float = datetime.fromisoformat(start_timestamp).timestamp()
        self._distributions = distributions or DEFAULT_DISTRIBUTIONS
        self._rng = np.random.default_rng(seed)

    # ── Tiempo actual ────────────────────────

    @property
    def current(self) -> float:
        """Tiempo lógico actual como Unix timestamp (float)."""
        return self._current

    def current_iso(self) -> str:
        """Tiempo lógico actual como string ISO 8601."""
        from datetime import datetime, timezone
        return datetime.fromtimestamp(self._current, tz=timezone.utc).isoformat()

    # ── Sampling y avance ────────────────────

    def sample_duration(self, activity_name: str) -> float:
        """
        Samplea una duración en segundos para la actividad dada.
        Usa la distribución log-normal configurada para esa actividad.
        Si la actividad no está en el diccionario, usa '_default'.
        """
        mu, sigma = self._distributions.get(
            activity_name,
            self._distributions["_default"]
        )
        duration = float(self._rng.lognormal(mean=mu, sigma=sigma))
        # Cap máximo de 7 días para evitar outliers extremos
        return min(duration, 7 * 24 * 3600)

    def advance(self, seconds: float) -> float:
        """
        Avanza el reloj en `seconds` segundos.
        Devuelve el nuevo timestamp.
        """
        self._current += seconds
        return self._current

    def advance_for_activity(self, activity_name: str) -> tuple[float, float]:
        """
        Samplea duración para la actividad y avanza el reloj.
        Devuelve (duration_seconds, new_timestamp).
        """
        duration = self.sample_duration(activity_name)
        new_ts = self.advance(duration)
        return duration, new_ts


# ─────────────────────────────────────────────────────────────────
# Utilidad: extrae distribuciones reales del BPIC
# Se llama una vez durante la fase de dataset analysis (días 15-28)
# y sobreescribe DEFAULT_DISTRIBUTIONS con valores reales.
# ─────────────────────────────────────────────────────────────────

def extract_bpic_distributions(xes_path: str) -> dict[str, tuple[float, float]]:
    """
    Lee un event log XES del BPIC y estima parámetros log-normal
    para cada actividad basándose en las duraciones reales.

    Args:
        xes_path: ruta al archivo .xes del BPIC 2012 o 2017

    Returns:
        dict {activity_name: (mu, sigma)} listo para pasar a SimulationClock
    """
    import pm4py
    import pandas as pd

    log = pm4py.read_xes(xes_path)
    df = pm4py.convert_to_dataframe(log)

    # Calcular duración de cada evento (diferencia con el siguiente del mismo caso)
    df = df.sort_values(["case:concept:name", "time:timestamp"])
    df["duration"] = (
        df.groupby("case:concept:name")["time:timestamp"]
        .diff()
        .shift(-1)
        .dt.total_seconds()
    )
    df = df.dropna(subset=["duration"])
    df = df[df["duration"] > 0]

    distributions: dict[str, tuple[float, float]] = {}
    for activity, group in df.groupby("concept:name"):
        durations = group["duration"].values
        if len(durations) < 5:
            continue
        # Fit log-normal: np.log transforma a distribución normal
        log_durations = np.log(durations)
        mu = float(np.mean(log_durations))
        sigma = float(np.std(log_durations))
        distributions[str(activity)] = (mu, sigma)

    distributions["_default"] = DEFAULT_DISTRIBUTIONS["_default"]
    return distributions
