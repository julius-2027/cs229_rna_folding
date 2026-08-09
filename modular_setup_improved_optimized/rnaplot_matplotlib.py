"""
rnaplot_matplotlib.py

Get RNA secondary structure layout coordinates from ViennaRNA's RNAplot
(running inside WSL, called via subprocess) and render the structure
natively in matplotlib -- so you get full control over styling and can
export to SVG, PNG, PDF, etc.

Strategy: RNAplot's "gml" (Graph Meta Language) output format is a
plain-text graph description with explicit per-nucleotide (X, Y)
coordinates and edges (backbone + base pairs). That's much easier
(and more robust across ViennaRNA versions) to parse than trying to
reverse SVG path/text elements back into coordinates.
"""

import re
import shlex
import subprocess
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class RNALayout:
    sequence: str
    x: list  # 1-indexed access via x[i-1]
    y: list
    pairs: list  # list of (i, j) 1-indexed base-pair tuples


def _run_rnaplot_gml(
    sequence: str,
    dotbracket: str,
    distro: str | None = None,
    wsl_workdir: str = "/tmp",
    layout_algorithm: int | None = None,
    timeout: int = 60,
) -> str:
    """Run RNAplot inside WSL with -o gml and return the raw text output."""
    if len(sequence) != len(dotbracket):
        raise ValueError(
            f"sequence (len={len(sequence)}) and dotbracket (len={len(dotbracket)}) "
            "must be the same length"
        )

    job_id = "jupyter_rnaplot_job"
    rna_input = f">{job_id}\n{sequence}\n{dotbracket}\n"

    # Use "gml" instead of "ssv": ssv isn't consistently supported/named
    # across ViennaRNA builds, but gml (Graph Meta Language) is available
    # everywhere and has a stable, well-documented text structure with
    # explicit node coordinates and edges.
    rnaplot_flags = ["-o", "gml"]
    if layout_algorithm is not None:
        rnaplot_flags += ["-t", str(layout_algorithm)]

    # Don't hardcode the output suffix -- glob for whatever file RNAplot
    # actually created for this job id, cat it, then clean up.
    inner_cmd = (
        f"cd {shlex.quote(wsl_workdir)} && "
        f"rm -f {shlex.quote(job_id)}_ss.* && "
        f"RNAplot {' '.join(rnaplot_flags)} 1>&2 && "
        f"f=$(ls {shlex.quote(job_id)}_ss.* 2>/dev/null | head -n1) && "
        f'if [ -z "$f" ]; then echo "no output file found" 1>&2; ls -la 1>&2; exit 1; fi && '
        f'cat "$f" && rm -f "$f"'
    )

    wsl_cmd = ["wsl"]
    if distro:
        wsl_cmd += ["-d", distro]
    wsl_cmd += ["--", "bash", "-lc", inner_cmd]

    result = subprocess.run(
        wsl_cmd,
        input=rna_input,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"RNAplot failed (exit code {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    if "graph" not in result.stdout or "node" not in result.stdout:
        raise RuntimeError(
            f"Unexpected RNAplot output, doesn't look like GML:\n{result.stdout[:1000]}\n"
            f"stderr:\n{result.stderr}"
        )

    return result.stdout


def _parse_gml(gml_text: str, sequence: str) -> RNALayout:
    """
    Parse RNAplot's Graph Meta Language (-o gml) text output.

    RNAplot emits one `node [ id N ... graphics [ x X y Y ] ]` block per
    nucleotide, in sequence order, followed by `edge [ source S target T ]`
    blocks -- both for the sequential backbone (S, S+1) and for base pairs
    (S, T with T far from S+1). We pull out (x, y) pairs and (source,
    target) pairs in the order they appear, rather than trying to fully
    parse GML's bracket nesting.
    """
    n = len(sequence)

    xy_pattern = re.compile(r"x\s+(-?[\d.]+)\s+y\s+(-?[\d.]+)")
    xy_matches = xy_pattern.findall(gml_text)
    if len(xy_matches) < n:
        raise RuntimeError(
            f"Expected {n} node coordinates in GML output, found {len(xy_matches)}."
        )
    x = [float(m[0]) for m in xy_matches[:n]]
    y = [float(m[1]) for m in xy_matches[:n]]

    edge_pattern = re.compile(r"source\s+(\d+)\s+target\s+(\d+)")
    pairs = []
    for s, t in edge_pattern.findall(gml_text):
        s, t = int(s), int(t)
        # Skip sequential backbone edges; keep the rest as base pairs.
        if abs(s - t) != 1:
            pairs.append((s, t))

    return RNALayout(sequence=sequence, x=x, y=y, pairs=pairs)


def get_rna_layout(
    sequence: str,
    dotbracket: str,
    distro: str | None = None,
    wsl_workdir: str = "/tmp",
    layout_algorithm: int | None = None,
) -> RNALayout:
    """Fetch nucleotide coordinates + base pairs for a structure via WSL RNAplot."""
    gml_text = _run_rnaplot_gml(
        sequence, dotbracket, distro=distro, wsl_workdir=wsl_workdir,
        layout_algorithm=layout_algorithm,
    )
    return _parse_gml(gml_text, sequence)


BASE_COLORS = {"A": "#7fc97f", "U": "#beaed4", "T": "#beaed4", "G": "#fdc086", "C": "#ffff99"}


def plot_rna_matplotlib(
    sequence: str,
    dotbracket: str,
    output_path: str,
    distro: str | None = None,
    wsl_workdir: str = "/tmp",
    layout_algorithm: int | None = None,
    figsize=(6, 6),
    dpi: int = 150,
    node_size: int = 220,
    show_labels: bool = True,
    highlight_positions: set[int] | None = None,
    highlight_color: str = "red",
) -> str:
    """
    Compute an RNA secondary structure layout via WSL RNAplot and render
    it as a native matplotlib figure, saved to `output_path`.

    The output format is inferred from the file extension (.svg, .png,
    .pdf, ... -- anything matplotlib's savefig supports). SVG/PDF are
    recommended if you want a scalable vector result; PNG if you just
    need a quick raster image.

    Parameters
    ----------
    sequence, dotbracket : str
        RNA sequence and matching dot-bracket structure.
    output_path : str
        Where to save the figure. Extension controls the format.
    distro, wsl_workdir, layout_algorithm :
        Passed through to RNAplot inside WSL (see get_rna_layout).
    figsize, dpi : matplotlib figure size / resolution.
    node_size : size of nucleotide markers.
    show_labels : whether to draw the nucleotide letters.
    highlight_positions : optional 1-indexed positions to draw in `highlight_color`.

    Returns
    -------
    str
        The output_path that was written.
    """
    layout = get_rna_layout(
        sequence, dotbracket, distro=distro, wsl_workdir=wsl_workdir,
        layout_algorithm=layout_algorithm,
    )
    highlight_positions = highlight_positions or set()

    fig, ax = plt.subplots(figsize=figsize)

    # Backbone (sequential connections)
    ax.plot(layout.x, layout.y, "-", color="black", linewidth=1.2, zorder=1)

    # Base pairs (dashed lines between paired nucleotides)
    for i, j in layout.pairs:
        ax.plot(
            [layout.x[i - 1], layout.x[j - 1]],
            [layout.y[i - 1], layout.y[j - 1]],
            "-", color="gray", linewidth=1.0, zorder=1,
        )

    # Nucleotides
    for pos in range(1, len(sequence) + 1):
        base = sequence[pos - 1]
        color = highlight_color if pos in highlight_positions else BASE_COLORS.get(base.upper(), "#cccccc")
        ax.scatter(
            layout.x[pos - 1], layout.y[pos - 1],
            s=node_size, facecolor=color, edgecolor="black",
            linewidth=0.8, zorder=2,
        )
        if show_labels:
            ax.text(
                layout.x[pos - 1], layout.y[pos - 1], base,
                ha="center", va="center", fontsize=8, zorder=3,
            )

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    seq = "GGGAAACCC"
    struct = "(((...)))"
    path = plot_rna_matplotlib(seq, struct, "structure.svg")
    print(f"Saved to {path}")
