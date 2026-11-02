"""Small, dependency-free regressions for the held-out benchmark selection."""
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("select_test.py")


def load_selector():
    if not SCRIPT.exists():
        raise AssertionError("The leakage-safe benchmark selector is not implemented.")
    spec = importlib.util.spec_from_file_location("benchmark_selection", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(identifier, symbols=("H", "H", "O"), signature="new"):
    numbers = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    return dict(structure_id=identifier, symbols=list(symbols), natoms=len(symbols),
                nelectron=sum(numbers[x] for x in symbols), charge=0, spin=0,
                coords_angstrom=[[float(i), 0., 0.] for i in range(len(symbols))],
                coordinate_signature_sha256=signature)


class SelectionTests(unittest.TestCase):
    def test_formula_overlap_is_rejected_in_each_split(self):
        selector = load_selector()
        for split in ("train", "validation", "pilot"):
            with self.subTest(split=split):
                exclusions = {"train": [row("a", ("H", "H"), "a")],
                              "validation": [row("b", ("F", "F"), "b")],
                              "pilot": [row("c", ("N", "N"), "c")]}
                exclusions[split] = [row("old", signature="old")]
                with self.assertRaisesRegex(ValueError, "formula"):
                    selector.select_rows([row("new")], exclusions,
                                         expected_ids=("new",))

    def test_id_or_geometry_overlap_is_rejected(self):
        selector = load_selector()
        for old in (row("new", ("H", "H"), "old"),
                    row("old", ("H", "H"), "new")):
            with self.subTest(old=old["structure_id"]):
                with self.assertRaisesRegex(ValueError, "ID|coordinate"):
                    selector.select_rows([row("new")], {"validation": [old]},
                                         expected_ids=("new",))

    def test_all_splits_are_combined_and_unseen_formula_is_allowed(self):
        selector = load_selector()
        candidate = row("new", ("C", "O", "O"))
        splits = {"train": [row("a", ("H", "H"), "a")],
                  "validation": [row("b", ("F", "F"), "b")],
                  "pilot": [row("c", ("N", "N"), "c")]}
        result = selector.select_rows([candidate], splits, expected_ids=("new",))
        self.assertEqual([r["structure_id"] for r in result], ["new"])
        self.assertEqual(result[0]["expected_nnao_nao"], 30)
        self.assertEqual(result[0]["formula"], "CO2")

    def test_old_aniline_composition_is_rejected(self):
        selector = load_selector()
        with self.assertRaisesRegex(ValueError, "aniline|formula"):
            selector.select_rows([row("new", tuple("C" * 6 + "H" * 7 + "N"))], {},
                                 expected_ids=("new",))

    def test_duplicate_source_id_and_missing_requested_id_are_rejected(self):
        selector = load_selector()
        for source in ([row("new"), row("new")], [row("different")]):
            with self.assertRaisesRegex(ValueError, "Duplicate|Missing"):
                selector.select_rows(source, {}, expected_ids=("new",))


if __name__ == "__main__":
    unittest.main()
