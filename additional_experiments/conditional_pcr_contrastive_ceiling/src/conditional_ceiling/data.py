"""Private C1B-H image access with explicit patient labels and strata.

Only the image tensor is returned to the model. HR, HER2, and assigned arm are
retained as CPU-side sampler metadata and encoded to an opaque stratum ID.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import stat
import struct
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd


IMAGE_SHAPE = (4, 7, 112, 176, 160)
CACHE_COLUMNS = (
    "patient_id",
    "cache_path",
    "cache_sha256",
    "cache_size_bytes",
    "cache_mtime_ns",
    "input_kind",
)
_VALIDATED_IMAGE_CACHE: set[tuple[str, int, int]] = set()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheRecord:
    patient_id: str
    path: Path
    sha256: str
    size_bytes: int
    mtime_ns: int


def load_cache_manifest(
    path: str | Path,
    expected_sha256: str,
    *,
    allowed_patient_ids: Iterable[str],
    verify_content: bool = False,
) -> dict[str, CacheRecord]:
    """Load the private C1B cache inventory and require full cohort coverage."""

    source = Path(path).expanduser().resolve(strict=True)
    if file_sha256(source) != expected_sha256:
        raise ValueError("C1B cache manifest SHA-256 mismatch")
    frame = pd.read_csv(source)
    if tuple(frame.columns) != CACHE_COLUMNS:
        raise ValueError("C1B cache manifest schema/order drifted")
    if frame["patient_id"].duplicated().any() or frame["cache_path"].duplicated().any():
        raise ValueError("C1B cache manifest repeats a patient or path")
    if set(frame["input_kind"].astype(str)) != {"c1b"}:
        raise ValueError("ceiling experiment accepts only C1B cache entries")
    allowed = {str(value) for value in allowed_patient_ids}
    indexed = frame.set_index(frame["patient_id"].astype(str), verify_integrity=True)
    missing = sorted(allowed - set(indexed.index))
    if missing:
        raise FileNotFoundError(f"C1B cache misses analysis patients: {missing[:5]}")
    output: dict[str, CacheRecord] = {}
    for patient_id in sorted(allowed):
        row = indexed.loc[patient_id]
        target = Path(str(row["cache_path"])).expanduser().resolve(strict=True)
        observed = target.stat()
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"C1B cache is not a regular file: {target}")
        size = int(row["cache_size_bytes"])
        mtime = int(row["cache_mtime_ns"])
        if observed.st_size != size or observed.st_mtime_ns != mtime:
            raise ValueError(f"C1B cache pinned stat changed for {patient_id}")
        digest = str(row["cache_sha256"])
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("cache digest must be lowercase SHA-256")
        if verify_content and file_sha256(target) != digest:
            raise ValueError(f"C1B cache digest mismatch for {patient_id}")
        output[patient_id] = CacheRecord(patient_id, target, digest, size, mtime)
    return output


def load_c1b_image(record: CacheRecord) -> np.ndarray:
    """Materialize only the model image member from a pinned C1B archive."""

    before = record.path.stat()
    if before.st_size != record.size_bytes or before.st_mtime_ns != record.mtime_ns:
        raise ValueError(f"cache changed before read: {record.patient_id}")
    with zipfile.ZipFile(record.path, "r") as archive:
        names = set(archive.namelist())
        if "image.npy" not in names or "patient_id.npy" not in names:
            raise ValueError("C1B archive lacks identity or image member")
        with archive.open("patient_id.npy") as stream:
            identity = np.load(stream, allow_pickle=False)
        if str(identity.item()) != record.patient_id:
            raise ValueError("C1B archive identity mismatch")
        member = archive.getinfo("image.npy")
        if member.compress_type != zipfile.ZIP_STORED:
            with archive.open("image.npy") as stream:
                image = np.load(stream, allow_pickle=False)
        else:
            # The frozen cache stores image.npy without compression.  Mapping
            # it directly avoids copying the 353 MB member once for validation
            # and again during collation on every stochastic revisit.
            with record.path.open("rb") as stream:
                stream.seek(member.header_offset)
                header = stream.read(30)
                if len(header) != 30:
                    raise ValueError("truncated C1B ZIP local header")
                signature, *_, filename_length, extra_length = struct.unpack(
                    "<IHHHHHIIIHH", header
                )
                if signature != 0x04034B50:
                    raise ValueError("invalid C1B ZIP local header")
                stream.seek(filename_length + extra_length, 1)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version in {(2, 0), (3, 0)}:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported C1B NPY member version")
                data_offset = stream.tell()
            image = np.memmap(
                record.path,
                dtype=dtype,
                mode="c",
                offset=data_offset,
                shape=shape,
                order="F" if fortran else "C",
            )
    after = record.path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("C1B cache changed during read")
    if image.shape != IMAGE_SHAPE or image.dtype != np.float32:
        raise ValueError(f"C1B image must be float32 {IMAGE_SHAPE}")
    validation_key = (str(record.path), record.mtime_ns, record.size_bytes)
    if validation_key not in _VALIDATED_IMAGE_CACHE:
        if not np.isfinite(image).all() or image.min() < -5.000001 or image.max() > 5.000001:
            raise ValueError("C1B image violates finite clipped input contract")
        _VALIDATED_IMAGE_CACHE.add(validation_key)
    return np.ascontiguousarray(image)


class CeilingImageDataset:
    """Patient-indexed dataset whose model-facing output has no clinical fields."""

    def __init__(
        self,
        patient_ids: Sequence[str],
        cache: Mapping[str, CacheRecord],
        labels: Mapping[str, int],
        stratum_ids: Mapping[str, int] | None = None,
        eligible_anchors: Iterable[str] = (),
    ) -> None:
        self.patient_ids = tuple(str(value) for value in patient_ids)
        if not self.patient_ids or len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("dataset patient IDs must be nonempty and unique")
        self.cache = cache
        self.labels = {str(key): int(value) for key, value in labels.items()}
        self.stratum_ids = {str(key): int(value) for key, value in (stratum_ids or {}).items()}
        self.eligible = {str(value) for value in eligible_anchors}
        for patient_id in self.patient_ids:
            if patient_id not in cache or patient_id not in self.labels:
                raise KeyError(f"dataset lacks cache/label for {patient_id}")
            if self.labels[patient_id] not in (0, 1):
                raise ValueError("pCR labels must be binary")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        patient_id = self.patient_ids[index]
        return {
            "patient_id": patient_id,
            "image": torch.from_numpy(load_c1b_image(self.cache[patient_id])),
            "label": torch.tensor(self.labels[patient_id], dtype=torch.long),
            "stratum_id": torch.tensor(self.stratum_ids.get(patient_id, -1), dtype=torch.long),
            "eligible_anchor": torch.tensor(patient_id in self.eligible, dtype=torch.bool),
        }


def collate_ceiling_batch(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    if not rows:
        raise ValueError("cannot collate an empty ceiling batch")
    eligible = torch.stack([row["eligible_anchor"] for row in rows])
    # The sampler always places its sole anchor first. Other rows are context
    # and must not become anchors merely because they are globally eligible.
    if len(rows) <= 4 and bool(eligible[0]):
        eligible = torch.zeros_like(eligible)
        eligible[0] = True
    return {
        "patient_id": [str(row["patient_id"]) for row in rows],
        "image": torch.stack([row["image"] for row in rows]),
        "label": torch.stack([row["label"] for row in rows]),
        "stratum_id": torch.stack([row["stratum_id"] for row in rows]),
        "eligible_anchor": eligible,
    }


class ConditionalStratumBatchSampler:
    """Yield exact-stratum anchor/positive/negative batches of at most four.

    Each stochastic epoch visits every eligible anchor exactly once in a
    deterministically reshuffled order. A batch contains one different same-pCR
    patient and one opposite-pCR patient from the exact same stratum. An optional
    fourth same-stratum patient improves GPU use without cross-stratum fallback.
    """

    def __init__(
        self,
        patient_ids: Sequence[str],
        stratum_ids: Mapping[str, int],
        labels: Mapping[str, int],
        *,
        seed: int,
        max_batch_size: int | None = 4,
        anchors_per_epoch: int | None = None,
    ) -> None:
        self.patient_ids = tuple(str(value) for value in patient_ids)
        self.seed = int(seed)
        self.epoch = 0
        self.max_batch_size = 4 if max_batch_size is None else int(max_batch_size)
        if self.max_batch_size < 3:
            raise ValueError("max_batch_size must be at least three")
        by_stratum: dict[int, list[int]] = {}
        for index, patient_id in enumerate(self.patient_ids):
            if patient_id not in stratum_ids or patient_id not in labels:
                raise KeyError("sampler metadata misses a patient")
            by_stratum.setdefault(int(stratum_ids[patient_id]), []).append(index)
        self.groups: dict[int, tuple[int, ...]] = {}
        self.labels_by_index = {
            index: int(labels[patient_id])
            for index, patient_id in enumerate(self.patient_ids)
        }
        self.eligible: list[tuple[int, int]] = []
        for _, indices in sorted(by_stratum.items()):
            stratum_id = int(stratum_ids[self.patient_ids[indices[0]]])
            classes = [self.labels_by_index[index] for index in indices]
            if set(classes) != {0, 1} or max(classes.count(0), classes.count(1)) < 2:
                continue
            self.groups[stratum_id] = tuple(indices)
            for anchor in indices:
                label = self.labels_by_index[anchor]
                same = [value for value in indices if value != anchor and self.labels_by_index[value] == label]
                opposite = [value for value in indices if self.labels_by_index[value] != label]
                if same and opposite:
                    self.eligible.append((stratum_id, anchor))
        if not self.eligible:
            raise ValueError("no usable exact matching stratum")
        self.anchors_per_epoch = (
            len(self.eligible) if anchors_per_epoch is None else int(anchors_per_epoch)
        )
        if not 1 <= self.anchors_per_epoch <= len(self.eligible):
            raise ValueError("anchors_per_epoch must be within the eligible-anchor count")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.anchors_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.eligible))[: self.anchors_per_epoch]
        for eligible_index in order:
            stratum_id, anchor = self.eligible[int(eligible_index)]
            group = self.groups[stratum_id]
            label = self.labels_by_index[anchor]
            same = [value for value in group if value != anchor and self.labels_by_index[value] == label]
            opposite = [value for value in group if self.labels_by_index[value] != label]
            positive = int(rng.choice(same))
            negative = int(rng.choice(opposite))
            selected = [anchor, positive, negative]
            # Prefer a second member of the negative class.  When the sampled
            # negative is itself globally eligible, this guarantees that its
            # explicit eligibility flag is also valid inside this physical
            # batch.  If that class is a singleton, the negative is globally
            # ineligible and any fourth member from the anchor class is safe.
            opposite_partner = [
                value
                for value in opposite
                if value != negative and value not in selected
            ]
            remaining = [value for value in group if value not in selected]
            if self.max_batch_size >= 4 and remaining:
                pool = opposite_partner or remaining
                selected.append(int(rng.choice(pool)))
            yield selected
