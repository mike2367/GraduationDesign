from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def nonempty_columns(
	rows: Sequence[Mapping[str, object]],
	*,
	always: Sequence[str] = (),
	never: Sequence[str] = (),
) -> list[str]:
	return [k for k in sorted(set(always) | {k for row in rows for k in row.keys() if k not in never}) if k not in never and any((v := r.get(k)) is not None and (not isinstance(v, str) or ((s := v.strip()) and s.lower() != "none")) for r in rows)]


def fmt_scalar(v: object) -> str:
	return "" if v is None else f"{v:.6g}" if isinstance(v, float) else str(v).strip()


def format_row_kv(
	row: Mapping[str, object],
	columns: Iterable[str],
	*,
	sep: str = " | ",
) -> str:
	return sep.join(f"{k}={fmt_scalar(v)}" for k in columns if (v := row.get(k)) is not None and (not isinstance(v, str) or ((s := v.strip()) and s.lower() != "none")))
