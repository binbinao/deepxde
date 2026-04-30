"""Smoke test for main.py: import + argparse only — no training in pytest."""
import importlib


def test_main_module_imports():
    mod = importlib.import_module("examples.radiation_slot.main")
    assert hasattr(mod, "build_argparser")
    assert hasattr(mod, "run_single_frequency")


def test_argparser_defaults():
    mod = importlib.import_module("examples.radiation_slot.main")
    parser = mod.build_argparser()
    args = parser.parse_args(["--mode", "single", "--freq", "15.0"])
    assert args.mode == "single"
    assert args.freq == 15.0
