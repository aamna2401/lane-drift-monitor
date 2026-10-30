PY ?= python
test:        ; $(PY) -m pytest -q tests
tune:        ; $(PY) scripts/tune.py --trials 60          # validation seeds only -> configs/*.json
eval:        ; $(PY) scripts/run_eval.py --clips 12        # test seeds -> outputs/results/clip_scores.csv
report:      ; $(PY) scripts/make_report.py                # tables + figures
real:        ; $(PY) scripts/eval_real_stills.py          # needs data/real_stills/*.jpg
all: test tune eval report
