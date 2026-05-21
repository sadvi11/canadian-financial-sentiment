"""Launch MLflow UI at http://localhost:5000"""
import subprocess, sys
print("Opening MLflow UI at http://localhost:5000")
print("Press Ctrl+C to stop")
subprocess.run([sys.executable, "-m", "mlflow", "ui",
    "--backend-store-uri", "mlruns", "--host", "0.0.0.0", "--port", "5000"])
