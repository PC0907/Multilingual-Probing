from __future__ import annotations

import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from study_truth_transport_v2.src.allocation import GroupRecord, build_overlap_plan, composition_quota


class AllocationTests(unittest.TestCase):
    def synthetic(self):
        rows = []
        for label in (0, 1):
            for stratum in ("a", "b", "c", "d"):
                for i in range(160):
                    rows.append(GroupRecord(f"{label}-{stratum}-{i}", label, stratum))
        return rows

    def test_exact_overlap_and_composition(self):
        plan = build_overlap_plan(self.synthetic(), repetitions=5, seed=7)
        self.assertEqual(len(plan["plans"]), 25)
        for row in plan["plans"]:
            self.assertEqual(len(row["source_group_ids"]), 400)
            self.assertEqual(len(row["target_group_ids"]), 400)
            self.assertEqual(row["shared_groups"], round(row["requested_overlap_fraction"] * 400))

    def test_reproducible(self):
        a = build_overlap_plan(self.synthetic(), repetitions=2, seed=9)
        b = build_overlap_plan(self.synthetic(), repetitions=2, seed=9)
        self.assertEqual(a, b)

    def test_sparse_stratum_is_excluded(self):
        rows = self.synthetic() + [GroupRecord("rare", 0, "singleton")]
        quota = composition_quota(rows)
        self.assertNotIn("0|singleton", quota)
        plan = build_overlap_plan(rows, repetitions=1)
        self.assertIn("rare", plan["excluded_groups"])

    def test_real_v2_training_groups(self):
        data = json.loads((ROOT / "generic_claims_2000_v3" / "data" / "en.json").read_text())
        members = defaultdict(list)
        for row in data:
            if row["partition"] == "train":
                members[row["group_id"]].append(row)
        records = []
        mixed = []
        for group_id, group in sorted(members.items()):
            group_labels = {r["label"] for r in group}
            if len(group_labels) != 1:
                mixed.append(group_id)
                continue
            label = group_labels.pop()
            # Minimal real-data feasibility test. Production plans use richer pair-specific strata.
            records.append(GroupRecord(group_id, label, "all"))
        self.assertEqual(len(mixed), 2)
        self.assertEqual(Counter(r.label for r in records), Counter({0: 580, 1: 590}))
        plan = build_overlap_plan(records, repetitions=2, seed=11)
        self.assertEqual(len(plan["plans"]), 10)


if __name__ == "__main__":
    unittest.main()
