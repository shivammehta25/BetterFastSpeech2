#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python fs2/train.py experiment=fs2_cormac_det trainer.devices=[5]
python fs2/train.py experiment=fs2_cormac_fm trainer.devices=[5]