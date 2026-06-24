"""
run_batch.py
------------
Corre la simulacion completa en batch y exporta el event log XES.

Uso:
    # Con casos sinteticos (sin BPIC):
    python run_batch.py --cases 50

    # Con casos reales del BPIC:
    python run_batch.py --cases 100 --bpic data/bpic/BPI_Challenge_2012.xes
"""

import argparse
import os
from dotenv import load_dotenv
from src.simulation_controller import SimulationController

load_dotenv()

def parse_args():
    p = argparse.ArgumentParser(description="Multi-LLM-Agent BPS — Batch Simulation")
    p.add_argument("--cases",   type=int, default=10,   help="Numero de casos a simular (default: 10)")
    p.add_argument("--bpic",    type=str, default=None, help="Ruta al archivo XES del BPIC (opcional)")
    p.add_argument("--model",   type=str, default=None, help="Modelo Ollama (default: variable de entorno)")
    p.add_argument("--seed",    type=int, default=42,   help="Semilla aleatoria (default: 42)")
    p.add_argument("--output",  type=str, default="output", help="Carpeta de salida (default: output/)")
    return p.parse_args()

def main():
    args = parse_args()

    model    = args.model or os.getenv("LLM_MODEL", "llama3.1")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    ctrl = SimulationController(
        n_cases=args.cases,
        model=model,
        ollama_base_url=base_url,
        bpic_xes_path=args.bpic,
        seed=args.seed,
        output_dir=args.output,
    )

    # Correr simulacion
    results = ctrl.run()

    # Exportar resultados
    ctrl.export_json("simulation.json")
    ctrl.export_xes("simulation.xes")

    # Resumen en consola
    df = ctrl.summary_dataframe()
    print("\n  Resumen por caso:")
    print(df.to_string(index=False))

    approved_pct = (df["status"] == "approved").mean() * 100
    print(f"\n  Tasa de aprobacion: {approved_pct:.1f}%")
    print(f"  Acciones promedio por caso: {df['n_actions'].mean():.1f}")
    print(f"  Tiempo promedio por caso: {df['duration_s'].mean():.1f}s\n")

if __name__ == "__main__":
    main()
