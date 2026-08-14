from __future__ import annotations

import torch

from conditional_ceiling.data import ConditionalStratumBatchSampler, collate_ceiling_batch


def test_sampler_is_one_cohort_pass_and_never_crosses_exact_strata() -> None:
    identifiers = tuple(f"P{index}" for index in range(12))
    strata = {patient: index // 6 for index, patient in enumerate(identifiers)}
    labels = {
        patient: (index % 6) // 3 for index, patient in enumerate(identifiers)
    }
    sampler = ConditionalStratumBatchSampler(
        identifiers, strata, labels, seed=2026, max_batch_size=4
    )
    assert len(sampler) == 12
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(batch[0] for batch in batches) == list(range(12))
    for batch in batches:
        assert 3 <= len(batch) <= 4
        assert len(set(batch)) == len(batch)
        assert len({strata[identifiers[index]] for index in batch}) == 1
        anchor = batch[0]
        same = [index for index in batch[1:] if labels[identifiers[index]] == labels[identifiers[anchor]]]
        opposite = [index for index in batch[1:] if labels[identifiers[index]] != labels[identifiers[anchor]]]
        assert same and opposite


def test_collator_marks_only_sampler_anchor() -> None:
    rows = []
    for index, label in enumerate((0, 0, 1, 1)):
        rows.append(
            {
                "patient_id": f"P{index}",
                "image": torch.zeros(4, 7, 2, 2, 2),
                "label": torch.tensor(label),
                "stratum_id": torch.tensor(3),
                "eligible_anchor": torch.tensor(True),
            }
        )
    batch = collate_ceiling_batch(rows)
    assert batch["eligible_anchor"].tolist() == [True, False, False, False]
