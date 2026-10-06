import numpy as np

from study_truth_transport_v2.runners.analyze_mmmlu_transfer import paired_accuracy
from study_truth_transport_v2.runners.prepare_mmmlu_external import selected_indices, wrong_index


def test_subject_selection_is_balanced_and_deterministic():
    subjects = ["a"] * 20 + ["b"] * 20
    first = selected_indices(subjects, 5, 4)
    second = selected_indices(subjects, 5, 4)
    assert first == second
    assert sum(index < 20 for index in first) == 5
    assert sum(index >= 20 for index in first) == 5


def test_wrong_choice_never_equals_correct():
    for row in range(100):
        for correct in range(4):
            assert wrong_index(row, correct, 3) != correct


def test_paired_accuracy_counts_ties_as_half():
    rows = [
        {"group_id": "a", "label": 0}, {"group_id": "a", "label": 1},
        {"group_id": "b", "label": 0}, {"group_id": "b", "label": 1},
    ]
    scores = np.asarray([0.0, 1.0, 2.0, 2.0])
    assert paired_accuracy(rows, scores) == 0.75
