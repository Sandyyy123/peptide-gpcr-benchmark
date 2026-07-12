"""Entry point: build demo data (if absent) and run the full benchmark."""
import os, subprocess, sys
HERE = os.path.dirname(__file__)
data = os.path.join(HERE, "data", "peptide_gpcr_demo.csv")
if not os.path.exists(data):
    subprocess.run([sys.executable, "-m", "src.make_dataset"], cwd=HERE, check=True)
subprocess.run([sys.executable, "-m", "src.benchmark", "--data", data], cwd=HERE, check=True)
