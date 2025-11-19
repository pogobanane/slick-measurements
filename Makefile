.PHONY: repl

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

WIDTH := 5.0
WIDTH2 := 5.5
DWIDTH := 11
DWIDTH2 := 13

repl:
	bpython -i <(echo 'import importlib.util; import sys; spec = importlib.util.spec_from_file_location("module.name", "${ROOT_DIR}/bpython.py"); repl = importlib.util.module_from_spec(spec); sys.modules["module.name"] = repl; spec.loader.exec_module(repl); repl.reload()')
	# bpython -i ${ROOT_DIR}/bpython.py


foobar.pdf:
	python3 foobar.py \
		-o foobar.pdf \
		--width $(WIDTH) --height 2 \
		--1 ./flake.nix

