#!/usr/bin/env bash
# Download the IU Chest X-ray report XMLs from NLM Open-i (text only, ~1 MB).
set -euo pipefail
mkdir -p data/raw
cd data/raw
curl -L -o NLMCXR_reports.tgz https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
tar -xzf NLMCXR_reports.tgz
echo "Reports extracted to data/raw/ecgen-radiology/ ($(ls ecgen-radiology | wc -l) files)"
