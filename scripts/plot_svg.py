#!/usr/bin/env python3
"""Render docs/scaling.svg from the strong-scaling tables in README.md."""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _parse_strong_tables(readme: str) -> dict[str, list[tuple[int, float, float]]]:
    series: dict[str, list[tuple[int, float, float]]] = {}
    current = None
    for line in readme.splitlines():
        if "Strong scaling, n = 1024" in line:
            current = "1024"
            series[current] = []
        elif "Strong scaling, n = 2048" in line:
            current = "2048"
            series[current] = []
        elif line.startswith("**Weak scaling"):
            current = None
        elif current and line.startswith("|"):
            cells = [c.strip().replace("*", "").replace(",", "") for c in line.strip("|").split("|")]
            if cells and cells[0].isdigit():
                ranks = int(cells[0])
                mlups = float(cells[3])
                speedup = float(cells[4].rstrip("×x"))
                series[current].append((ranks, mlups, speedup))
    return series


def render(readme_path: pathlib.Path, out_path: pathlib.Path) -> None:
    series = _parse_strong_tables(readme_path.read_text(encoding="utf-8"))
    s1024 = series["1024"]
    s2048 = series["2048"]

    w, h = 920, 380
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" rx="12" fill="#0d1117"/>',
        '<text x="32" y="36" fill="#f0f6fc" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700">'
        'MPI 2D Jacobi Strong Scaling: Cache-Friendly (1024²) vs Bandwidth-Bound (2048²)</text>',
        '<text x="32" y="58" fill="#8b949e" font-family="Segoe UI, Arial, sans-serif" font-size="12">'
        'Intel i7-14650HX (WSL2, 12 threads, OpenMPI 5.0.11) · same binary, 5 reads + 1 write per cell</text>',
    ]

    # Left panel: Speedup curve vs Ideal
    x0, y0, pw, ph = 32, 84, 420, 266
    lines.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="8" fill="#161b22" stroke="#30363d"/>')
    lines.append(
        f'<text x="{x0 + 16}" y="{y0 + 26}" fill="#e6edf3" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="13" font-weight="600">Speedup vs Ranks (ideal = linear)</text>'
    )
    gx, gy, gw, gh = x0 + 52, y0 + 48, pw - 86, ph - 94

    def sx(ranks: int) -> float:
        return gx + (ranks - 1) / 7.0 * gw

    def sy(sp: float) -> float:
        return gy + gh - (sp / 8.0) * gh

    for tick in (1, 2, 4, 6, 8):
        ty = sy(float(tick))
        lines.append(f'<line x1="{gx}" y1="{ty:.1f}" x2="{gx + gw}" y2="{ty:.1f}" stroke="#21262d"/>')
        lines.append(
            f'<text x="{gx - 8}" y="{ty + 4:.1f}" text-anchor="end" fill="#8b949e" '
            f'font-family="Consolas, monospace" font-size="11">{tick}×</text>'
        )
    for r in (1, 2, 4, 8):
        tx = sx(r)
        lines.append(
            f'<text x="{tx:.1f}" y="{gy + gh + 18}" text-anchor="middle" fill="#8b949e" '
            f'font-family="Consolas, monospace" font-size="11">{r} ranks</text>'
        )

    lines.append(
        f'<line x1="{sx(1):.1f}" y1="{sy(1):.1f}" x2="{sx(8):.1f}" y2="{sy(8):.1f}" '
        f'stroke="#484f58" stroke-dasharray="5,5" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="{sx(6.8):.1f}" y="{sy(7.2):.1f}" fill="#8b949e" font-family="Consolas, monospace" font-size="10">ideal</text>'
    )

    for label, pts, color, dy in [
        ("n=1024 (8 MB/arr, in L3)", s1024, "#56d364", -10),
        ("n=2048 (32 MB/arr, DRAM)", s2048, "#f0883e", 16),
    ]:
        poly = " ".join(f"{sx(r):.1f},{sy(sp):.1f}" for r, _, sp in pts)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{poly}"/>')
        for r, _, sp in pts:
            lines.append(f'<circle cx="{sx(r):.1f}" cy="{sy(sp):.1f}" r="4" fill="{color}"/>')
        lr, _, lsp = pts[-1]
        lines.append(
            f'<text x="{sx(lr) - 6:.1f}" y="{sy(lsp) + dy:.1f}" text-anchor="end" fill="{color}" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="700">{sp_lbl(label, lsp)}</text>'
        )

    # Right panel: Throughput (MLUPS) bars
    rx0, ry0, rpw, rph = 472, 84, 416, 266
    lines.append(f'<rect x="{rx0}" y="{ry0}" width="{rpw}" height="{rph}" rx="8" fill="#161b22" stroke="#30363d"/>')
    lines.append(
        f'<text x="{rx0 + 16}" y="{ry0 + 26}" fill="#e6edf3" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="13" font-weight="600">Throughput (MLUPS) — 8 ranks saturate memory bus</text>'
    )
    max_mlups = max(m for _, m, _ in s1024)
    bx0, bw_max = rx0 + 74, rpw - 168
    for idx, ((r1, m1, _), (_, m2, _)) in enumerate(zip(s1024, s2048)):
        by = ry0 + 52 + idx * 50
        lines.append(
            f'<text x="{bx0 - 10}" y="{by + 20}" text-anchor="end" fill="#c9d1d9" '
            f'font-family="Consolas, monospace" font-size="11">{r1} rank{"s" if r1 > 1 else ""}</text>'
        )
        w1 = max(4.0, (m1 / max_mlups) * bw_max)
        w2 = max(4.0, (m2 / max_mlups) * bw_max)
        lines.append(f'<rect x="{bx0}" y="{by}" width="{w1:.1f}" height="14" rx="3" fill="#56d364"/>')
        lines.append(
            f'<text x="{bx0 + w1 + 6:.1f}" y="{by + 11}" fill="#56d364" font-family="Consolas, monospace" font-size="10">'
            f'1024²: {m1:,.0f}</text>'
        )
        lines.append(f'<rect x="{bx0}" y="{by + 18}" width="{w2:.1f}" height="14" rx="3" fill="#f0883e"/>')
        lines.append(
            f'<text x="{bx0 + w2 + 6:.1f}" y="{by + 29}" fill="#f0883e" font-family="Consolas, monospace" font-size="10">'
            f'2048²: {m2:,.0f}</text>'
        )

    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


def sp_lbl(label: str, sp: float) -> str:
    short = re.sub(r"\s*\(.*?\)", "", label)
    return f"{short}: {sp:.2f}×"


if __name__ == "__main__":
    render(ROOT / "README.md", ROOT / "docs" / "scaling.svg")
