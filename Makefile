.PHONY: repl

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

repl:
	bpython -i <(echo 'import importlib.util; import sys; spec = importlib.util.spec_from_file_location("module.name", "${ROOT_DIR}/bpython.py"); repl = importlib.util.module_from_spec(spec); sys.modules["module.name"] = repl; spec.loader.exec_module(repl); repl.reload()')
	# bpython -i ${ROOT_DIR}/bpython.py




