"""Smoke test for main_scan.py: import + argparse only."""
import importlib


def test_main_scan_imports():
    mod = importlib.import_module("examples.radiation_slot.main_scan")
    assert hasattr(mod, "build_argparser")
    assert hasattr(mod, "run_scan")
    assert hasattr(mod, "plot_s_param_curve")


def test_main_scan_argparser_defaults():
    mod = importlib.import_module("examples.radiation_slot.main_scan")
    parser = mod.build_argparser()
    args = parser.parse_args([])
    assert args.f_start == 12.0
    assert args.f_stop == 18.0
    assert args.f_step == 0.5
    assert args.pinn_freq == 15.0
