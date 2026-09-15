.PHONY: test paper-outputs

test:
	python -m pytest

paper-outputs:
	python scripts/build_paper_outputs.py
