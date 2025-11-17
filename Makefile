# Makefile for training and running the Prompt Injection Detector
# Adjust python path or environment as needed

PYTHON=python3
SCRIPT=promptBert.py

.PHONY: all train run clean

all: train

train:
	$(PYTHON) $(SCRIPT)

run:
	$(PYTHON) $(SCRIPT)

clean:
	rm -rf __pycache__
	rm -rf trainedDetector
	rm -f *.pyc
