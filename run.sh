#!/bin/bash
# Starts the dashboard. Usage:  ./run.sh   (then open http://localhost:8501)
cd "$(dirname "$0")"
~/micromamba/envs/rx/bin/streamlit run app.py
