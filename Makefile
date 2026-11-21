.PHONY: help data train report serve test clean

PYTHON ?= python3
DATA ?= data/synthetic
OUT ?= artifacts/avf-demo
PATIENTS ?= 300
EPOCHS ?= 14

help:
	@echo "make data     generate + preprocess the synthetic cohort"
	@echo "make train    train the two-stream model"
	@echo "make report   render a demo HTML report"
	@echo "make serve    run the FastAPI demo on :8000"
	@echo "make test     run unit tests"

data:
	PYTHONPATH=src $(PYTHON) -m avfsentinel.synth --out $(DATA) --patients $(PATIENTS)
	PYTHONPATH=src $(PYTHON) -m avfsentinel.prepare --data $(DATA)

train:
	PYTHONPATH=src $(PYTHON) -m avfsentinel.train --data $(DATA) --out $(OUT) \
		--epochs $(EPOCHS) --batch-size 12 --freeze-epochs 2

report:
	PYTHONPATH=src $(PYTHON) -m avfsentinel.infer --checkpoint $(OUT)/best.pt \
		--demo --scenario thrombosis --data $(DATA) --out $(OUT) \
		--report $(OUT)/demo_report.html

serve:
	PYTHONPATH=src $(PYTHON) -m avfsentinel.serve --checkpoint $(OUT)/best.pt --data $(DATA)

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf $(DATA) $(OUT) artifacts/reports
