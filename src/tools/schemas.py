"""
schemas.py
----------
Schemas Pydantic para las tools de cada agente.

Regla clave: cada tool representa UNA acción atómica del proceso real.
El agente LLM no puede inventar acciones fuera de este conjunto.
Esto es el mecanismo de control de alucinaciones de la tesis.

Cada schema hereda de BaseModel de Pydantic v2.
LangChain los convierte automáticamente en function-calling schemas
para la API de OpenAI/Anthropic.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums compartidos
# ─────────────────────────────────────────────

class DocumentStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FRAUDULENT = "fraudulent"


class CaseDecision(str, Enum):
    FORWARD = "forward"
    RETURN_FOR_REVISION = "return_for_revision"
    REJECT = "reject"


# ─────────────────────────────────────────────
# Tools del Junior Clerk
# ─────────────────────────────────────────────

class IntakeApplication(BaseModel):
    """
    Registra la recepción inicial de una solicitud de préstamo.
    Primera acción obligatoria del Junior Clerk en cada caso.
    """
    case_id: str = Field(description="ID del caso de préstamo")
    applicant_acknowledged: bool = Field(
        description="¿Se ha informado al solicitante de la recepción?"
    )
    initial_notes: str = Field(
        description="Observaciones iniciales del clerk sobre el caso",
        max_length=500,
    )


class CheckDocuments(BaseModel):
    """
    Verifica que la documentación presentada está completa y es válida.
    """
    case_id: str
    document_status: DocumentStatus
    missing_documents: list[str] = Field(
        default_factory=list,
        description="Lista de documentos faltantes si document_status == 'incomplete'",
    )
    notes: str = Field(default="", max_length=500)


class ForwardCase(BaseModel):
    """
    Envía el caso al Senior Clerk para validación profunda.
    Solo válido si document_status fue 'complete'.
    """
    case_id: str
    forwarded_to: str = Field(
        default="senior_clerk",
        description="Destino del caso (siempre 'senior_clerk' en esta versión)"
    )
    priority: str = Field(
        description="'normal' | 'high' según criterio del clerk",
        pattern="^(normal|high)$",
    )


# ─────────────────────────────────────────────
# Tools del Senior Clerk
# ─────────────────────────────────────────────

class ValidateApplication(BaseModel):
    """
    Realiza la validación profunda de la solicitud.
    Verifica coherencia entre documentos, ingresos y monto solicitado.
    """
    case_id: str
    income_verified: bool
    debt_to_income_ratio: float = Field(
        ge=0.0,
        description="Ratio deuda/ingreso calculado (cuota_mensual / ingreso_mensual). Puede superar 1.0 en casos extremos."
    )
    validation_notes: str = Field(max_length=800)


class RequestAdditionalInfo(BaseModel):
    """
    Solicita información adicional al solicitante o al Junior Clerk.
    Activa un loop de revisión en el grafo.
    """
    case_id: str
    requested_from: str = Field(
        description="'applicant' | 'junior_clerk'",
        pattern="^(applicant|junior_clerk)$",
    )
    information_needed: list[str] = Field(
        description="Lista específica de información requerida"
    )
    reason: str = Field(max_length=500)


class EscalateCase(BaseModel):
    """
    Escala el caso al Credit Officer para decisión final.
    """
    case_id: str
    risk_summary: str = Field(
        description="Resumen del perfil de riesgo del solicitante",
        max_length=800,
    )
    recommendation: str = Field(
        description="'approve' | 'reject' | 'conditional'",
        pattern="^(approve|reject|conditional)$",
    )


# ─────────────────────────────────────────────
# Tools del Credit Officer
# ─────────────────────────────────────────────

class AssessRisk(BaseModel):
    """
    Evaluación formal de riesgo crediticio.
    Considera credit score, ratio deuda/ingreso y monto solicitado.
    """
    case_id: str
    credit_score: int = Field(ge=300, le=850)
    risk_category: str = Field(
        description="'low' | 'medium' | 'high'",
        pattern="^(low|medium|high)$",
    )
    risk_factors: list[str] = Field(
        description="Factores de riesgo identificados"
    )
    assessment_notes: str = Field(max_length=800)


class ApproveLoan(BaseModel):
    """
    Aprueba la solicitud de préstamo con condiciones específicas.
    """
    case_id: str
    approved_amount: float = Field(gt=0)
    interest_rate: float = Field(
        ge=0.0, le=1.0,
        description="Tasa de interés anual (ej: 0.045 para 4.5%)"
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Condiciones adicionales de la aprobación"
    )
    approval_notes: str = Field(default="", max_length=500)


class RejectLoan(BaseModel):
    """
    Rechaza la solicitud de préstamo con justificación formal.
    """
    case_id: str
    rejection_reasons: list[str] = Field(
        min_length=1,
        description="Al menos una razón formal de rechazo"
    )
    rejection_notes: str = Field(max_length=500)


class ReturnApplicationEarly(BaseModel):
    """
    Devuelve la solicitud al solicitante sin procesar.
    Usado por Junior Clerk cuando el caso no cumple requisitos mínimos:
    documentación fraudulenta, score < 480, o ratio deuda/ingreso > 0.70.
    """
    case_id: str
    return_reason: str = Field(
        description="'incomplete_docs' | 'fraudulent_docs' | 'below_minimum_score' | 'excessive_ratio'",
        pattern="^(incomplete_docs|fraudulent_docs|below_minimum_score|excessive_ratio)$",
    )
    details: str = Field(max_length=400)


# ─────────────────────────────────────────────
# Registro de tools por agente
# Usado por los nodos LangGraph para instanciar el LLM con tools correctas
# ─────────────────────────────────────────────

AGENT_TOOLS: dict[str, list[type[BaseModel]]] = {
    "junior_clerk":   [IntakeApplication, CheckDocuments, ForwardCase, ReturnApplicationEarly],
    "senior_clerk":   [ValidateApplication, RequestAdditionalInfo, EscalateCase],
    "credit_officer": [AssessRisk, ApproveLoan, RejectLoan],
}
