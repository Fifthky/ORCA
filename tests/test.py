"""Dependency-free tests for CLI, module boundaries and execution controls."""

import argparse
import ast
import builtins
import contextlib
import csv
import io
import math
from pathlib import Path
import symtable
import sys
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cli import build_parser, _expand_refiner_variants, _preflight_validate_and_print_plan
from model_registry import normalize_model_name, resolve_model_path


def definitions(filename, names, namespace=None):
    """Load pure functions without importing GPU or foundation-model packages."""
    tree = ast.parse((ROOT / filename).read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(body) != len(names) + 1:
        raise AssertionError(f"Definitions not found in {filename}: {names}")
    scope = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), filename, "exec"), scope)
    return scope


def parsed(*flags):
    args = build_parser().parse_args(list(flags))
    args._command_line = " ".join(flags)
    with contextlib.redirect_stdout(io.StringIO()):
        _preflight_validate_and_print_plan(args, selected_refiners=args.refiner)
    return args


class Tests(unittest.TestCase):
    def test_all_sources_compile(self):
        for path in ROOT.rglob("*.py"):
            compile(path.read_text(), str(path), "exec")

    def test_no_unbound_globals(self):
        for path in ROOT.rglob("*.py"):
            if "tests" in path.parts:
                continue
            top = symtable.symtable(path.read_text(), str(path), "exec")
            defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
            defined |= set(dir(builtins)) | {"__file__", "__name__", "__package__"}
            def check(table):
                for symbol in table.get_symbols():
                    if symbol.is_global() and symbol.is_referenced():
                        self.assertIn(symbol.get_name(), defined, f"{path}: {table.get_name()}")
                for child in table.get_children():
                    check(child)
            check(top)

    def test_local_imports_and_cycles(self):
        paths = {str(p.relative_to(ROOT).with_suffix("")).replace("/", "."): p
                 for p in ROOT.rglob("*.py") if "tests" not in p.parts}
        symbols = {}
        graphs = {}
        for module, path in paths.items():
            top = symtable.symtable(path.read_text(), str(path), "exec")
            symbols[module] = {s.get_name() for s in top.get_symbols()}
            graphs[module] = set()
        for module, path in paths.items():
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    root = node.module.split(".")[0]
                    if root not in {"core", "eval", "data", "cli", "run", "reporting", "model_registry"}:
                        continue
                    self.assertIn(node.module, paths, f"{module}: missing {node.module}")
                    graphs[module].add(node.module)
                    for alias in node.names:
                        self.assertIn(alias.name, symbols[node.module], f"{module}: missing {alias.name}")
        def visit(node, ancestors):
            self.assertNotIn(node, ancestors, f"Import cycle: {ancestors} -> {node}")
            for child in graphs[node]:
                visit(child, ancestors + [node])
        for node in graphs:
            visit(node, [])

    def test_input_modes_and_update_rules(self):
        args = parsed("--refiner", "ORCA", "--refiner_input", "all", "xy", "x", "y", "e_past",
                      "--update_rule", "plain", "bayesian")
        variants = _expand_refiner_variants("ORCA", args)
        self.assertEqual(len(variants), 10)
        self.assertEqual({v["refiner_input"] for v in variants}, {"all", "xy", "x", "y", "e_past"})

    def test_main_suffix(self):
        args = parsed("--refiner", "ORCA", "--refiner_input", "xy", "--update_rule", "bayesian", "--random_seed", "42")
        self.assertEqual(_expand_refiner_variants("ORCA", args)[0]["variant_suffix"], "xy_bayesian_buf3000_batch256_seed42")

    def test_gate_channel_and_router_options(self):
        args = parsed("--refiner", "ORCA", "--refiner_input", "xy", "--router", "hard",
                      "--no-channel_mix", "--force_gate_open")
        variant = _expand_refiner_variants("ORCA", args)[0]
        self.assertTrue(variant["variant_suffix"].endswith("hard_gate_open_ci"))
        args = parsed("--refiner", "ORCA", "--router", "inema")
        self.assertEqual(_expand_refiner_variants("ORCA", args)[0]["router"], "inema")

    def test_batch_only_for_baselines(self):
        for name in ["ORCA", "ORCA_NoDecay", "ELF", "Ridge", "ETS"]:
            with self.assertRaises(ValueError):
                parsed("--refiner", name, "--training_method", "batch")
        args = parsed("--refiner", "TAFAS", "--training_method", "batch")
        self.assertEqual(_expand_refiner_variants("TAFAS", args)[0]["training_method"], "batch")

    def test_removed_options_rejected(self):
        for flags in [["--attn_maps"], ["--cahce"], ["--bay_loss", "mae"], ["--batch", "256"],
                      ["--routing_temperature", "0.1", "0.2"]]:
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                build_parser().parse_args(flags)
        for rule in ["semi_prior", "prior"]:
            with self.assertRaises(ValueError):
                parsed("--update_rule", rule)

    def test_backbone_path(self):
        scope = definitions("run.py", ["_configure_model_args"], {
            "argparse": argparse, "Path": Path, "normalize_model_name": normalize_model_name,
            "resolve_model_path": resolve_model_path})
        args = build_parser().parse_args(["--tsfm_local_path", "/weights/custom", "--tsfm_model_prefix", "/weights"])
        self.assertEqual(scope["_configure_model_args"](args, "chronos-2").tsfm_local_path, "/weights/custom")
        args.tsfm_local_path = None
        self.assertEqual(scope["_configure_model_args"](args, "chronos-2").tsfm_local_path, "/weights/chronos-2")

    def test_force_download(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dataset.csv"
            path.touch()
            download = Mock(return_value=path)
            scope = definitions("run.py", ["_resolve_csv_path"], {
                "Path": Path, "resolve_cached_csv_path": lambda *a, **k: path, "ensure_dataset_csv": download})
            args = build_parser().parse_args(["--dataset", "ETTh1", "--auto_download", "--force_download"])
            self.assertEqual(scope["_resolve_csv_path"](args, "ETTh1"), path.resolve())
            download.assert_called_once()
            self.assertTrue(download.call_args.kwargs["force"])
            download.reset_mock()
            args.force_download = False
            scope["_resolve_csv_path"](args, "ETTh1")
            download.assert_not_called()

    def test_cache_filename_stable(self):
        scope = definitions("eval/inference_cache.py", ["_build_infer_cache_path"], {"Path": Path})
        args = argparse.Namespace(model="timesfm-2.5", context_length=520, pred_len=96)
        path = scope["_build_infer_cache_path"](args, "ETTh1")
        self.assertEqual(path.name, "timesfm_2_ETTh1_ctx520_pred96_s1_npyv3")

    def test_tirex_online_rejected_without_environment_changes(self):
        scope = definitions("eval/model_backends.py", ["resolve_model_ref"], {
            "Path": Path, "HF_MODEL_IDS": {"tirex": "NX-AI/TiRex-1.1-gifteval"}, "BackendPredictorError": RuntimeError})
        with self.assertRaisesRegex(RuntimeError, "local checkpoint"):
            scope["resolve_model_ref"]("tirex", "/weights/tirex", True)
        self.assertEqual(scope["resolve_model_ref"]("tirex", "/weights/tirex", False), "/weights/tirex")
        self.assertNotIn("os.environ", (ROOT / "eval/model_backends.py").read_text())

    def test_chronos_preserves_device_index(self):
        base = type("TestPredictorBase", (), {"__init__": lambda self, **kwargs: None})
        pipeline = SimpleNamespace(from_pretrained=Mock(return_value=object()))
        chronos = SimpleNamespace(BaseChronosPipeline=pipeline, Chronos2Pipeline=pipeline)
        scope = definitions("eval/model_backends.py", ["Chronos2Predictor"], {
            "_PredictorBase": base, "torch": SimpleNamespace(bfloat16="bfloat16"),
            "BackendPredictorError": RuntimeError})
        with patch.dict(sys.modules, {"chronos": chronos}):
            scope["Chronos2Predictor"]("/weights/chronos", 96, 520, 1, "cuda:1")
        self.assertEqual(pipeline.from_pretrained.call_args.kwargs["device_map"], "cuda:1")

    def test_failure_and_success_status(self):
        namespace = {"Path": Path, "sys": sys, "traceback": traceback,
                     "_metric_dict_with_nan": lambda: {"MSE[mean]": float("nan")}}
        namespace.update(definitions("reporting.py", ["_build_failed_eval_record"], namespace))
        namespace["run_csv_evaluation"] = Mock(return_value={"agg_metrics_base": {"MSE[mean]": 1.0}})
        scope = definitions("run.py", ["_execute_configuration", "_raise_for_failed_runs"], namespace)
        args = argparse.Namespace(csv_path="ETTh1.csv", pred_len=96)
        variant = {"refiner_tag": "ORCA", "variant_suffix": "xy_bayesian"}
        result = scope["_execute_configuration"](args, None, dataset_label="ETTh1", model_short_name="chronos-2", refiner="ORCA", variant=variant)
        self.assertEqual(result["agg_metrics_base"]["MSE[mean]"], 1.0)
        scope["_raise_for_failed_runs"]([result])
        scope["run_csv_evaluation"] = Mock(side_effect=RuntimeError("test failure"))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            failed = scope["_execute_configuration"](args, None, dataset_label="ETTh1", model_short_name="chronos-2", refiner="ORCA", variant=variant)
            self.assertTrue(math.isnan(failed["agg_metrics_base"]["MSE[mean]"]))
            with self.assertRaises(SystemExit) as caught:
                scope["_raise_for_failed_runs"]([result, failed])
        self.assertEqual(caught.exception.code, 1)

    def test_horizon_average_and_missing_horizon(self):
        scope = definitions("reporting.py", ["_build_pred_len_average_records", "_average_metric_dicts_strict",
                            "_metric_or_nan", "_to_float_or_nan", "_strict_mean_or_nan", "_metric_dict_with_nan"],
                            {"math": math, "CORE_METRIC_KEYS": ["MSE[mean]"], "first_value": float})
        records = [{"dataset_label": "ETTh1", "model_short_name": "chronos-2", "refiner": "ORCA",
                    "refiner_tag": "ORCA", "variant_suffix": "xy_bayesian_seed42", "training_method": "online",
                    "refiner_input": "xy", "update_rule": "bayesian", "pred_len": h,
                    "agg_metrics_base": {"MSE[mean]": base}, "agg_metrics_refined": {"MSE[mean]": adapted}}
                   for h, base, adapted in [(30, 0.2, 0.1), (96, 0.4, 0.3), (336, 0.6, 0.5)]]
        result = scope["_build_pred_len_average_records"](records, [30, 96, 336])[0]
        self.assertAlmostEqual(result["agg_metrics_base"]["MSE[mean]"], 0.4)
        self.assertAlmostEqual(result["agg_metrics_refined"]["MSE[mean]"], 0.3)
        missing = scope["_build_pred_len_average_records"](records[:2], [30, 96, 336])[0]
        self.assertTrue(math.isnan(missing["agg_metrics_refined"]["MSE[mean]"]))

    def test_summary_csv_resume_roundtrip(self):
        names = ["_summary_formal_model_name", "_summary_to_float_nan", "_summary_format_metric",
                 "_summary_format_change", "summary_pct_change", "_summary_mean_or_nan",
                 "write_split_summary_metric_csv", "parse_split_summary_metric_csv", "load_existing_split_summary_records"]
        scope = definitions("eval/eval_util.py", names, {"math": math, "csv": csv, "Path": Path, "first_value": float})
        records = [{"dataset_label": "ETTh1", "model_short_name": "timesfm-2.5",
                    "agg_metrics_base": {"MSE[mean]": 0.4}, "agg_metrics_refined": {"MSE[mean]": 0.3}}]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.csv"
            scope["write_split_summary_metric_csv"](csv_path=path, dataset_order=["ETTh1"],
                model_order=["timesfm-2.5"], records=records, metric_key="MSE[mean]")
            self.assertIn("-25.0%", path.read_text())
            self.assertIn("TimesFM-2.5", path.read_text())
            restored, complete, datasets, models = scope["load_existing_split_summary_records"](
                mae_csv_path=path, mse_csv_path=path, pred_len=96, refiner="ORCA", refiner_tag="ORCA",
                variant_suffix="xy_bayesian", training_method="online", refiner_input="xy", update_rule="bayesian",
                online_buffer_windows=3000, mae_metric_key="MAE[mean]", mse_metric_key="MSE[mean]")
            self.assertEqual(complete, {("ETTh1", "timesfm-2.5")})
            self.assertEqual(datasets, ["ETTh1"])
            self.assertEqual(models, ["timesfm-2.5"])
            self.assertEqual(restored[0]["agg_metrics_refined"]["MSE[mean]"], 0.3)


if __name__ == "__main__":
    unittest.main()
