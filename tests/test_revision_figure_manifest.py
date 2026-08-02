import hashlib
import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "runs/revision_1/revision_figure_manifest.json"
MANUSCRIPT_FIGURE_DIR = REPO_ROOT.parent / "LiteViLNetPaperRAL" / "figures"
VAL_SPLIT = REPO_ROOT / "configs/splits/kitti_road/stratified_seed20260723/val.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RevisionFigureManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(MANIFEST.read_text())

    def test_every_recorded_output_exists_and_matches_hash(self):
        self.assertGreaterEqual(len(self.payload["outputs"]), 7)
        for record in self.payload["outputs"]:
            path = Path(record["path"])
            self.assertTrue(path.is_file(), path)
            self.assertFalse(path.is_relative_to(MANUSCRIPT_FIGURE_DIR), path)
            self.assertEqual(path.stat().st_size, record["bytes"])
            self.assertEqual(sha256(path), record["sha256"])

    def test_qualitative_samples_are_fixed_validation_examples(self):
        qualitative = self.payload["details"]["qualitative"]
        selected = qualitative["samples"]
        selected_ids = {record["sample_id"] for record in selected}
        validation_ids = {
            line.strip().removesuffix(".png")
            for line in VAL_SPLIT.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(len(selected), 4)
        self.assertTrue(selected_ids <= validation_ids)
        self.assertTrue({"UM", "UMM", "UU"} <= {record["category"] for record in selected})
        self.assertEqual(len({record["adi_sha256"] for record in selected}), 4)
        self.assertEqual(qualitative["threshold"], 0.66)
        self.assertEqual(qualitative["checkpoint_epoch"], 45)
        self.assertEqual(
            qualitative["checkpoint_sha256"],
            "49f07b83fdad95dc7330c0d568e38532dcb2f748ef117e89a95ac824d957fa79",
        )
        for record in selected:
            assets = record["manual_drawing_assets"]
            self.assertEqual(
                set(assets),
                {
                    "rgb",
                    "stored_adi",
                    "prediction_overlay",
                    "binary_prediction",
                    "probability",
                    "error_map",
                },
            )
            self.assertTrue(all(Path(path).is_file() for path in assets.values()))

    def test_g1_depth_and_segmentation_contents_are_in_correct_order(self):
        robot = self.payload["details"]["robot"]
        self.assertGreater(
            robot["after"]["depth_labeled_black_fraction"],
            robot["after"]["segmentation_labeled_black_fraction"],
        )

    def test_diagram_contracts_name_the_implemented_paths(self):
        architecture = self.payload["details"]["architecture"]["contract"]
        msfm = self.payload["details"]["msfm"]["contract"]
        for term in ("MobileNetV3", "Conv stem", "DSConv", "MSFM", "LKB"):
            self.assertIn(term, architecture)
        for term in ("pooled RGB query", "spatial ADI K/V", "B×1×N_l", "gate blend"):
            self.assertIn(term, msfm)


if __name__ == "__main__":
    unittest.main()
