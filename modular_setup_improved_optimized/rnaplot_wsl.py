"""
rnaplot_wsl.py

Run ViennaRNA's RNAplot inside WSL from a Windows Python/Jupyter process,
using subprocess, and save the resulting 2D structure drawing as an SVG.
"""

import subprocess
import shlex


def rnaplot_svg(
    sequence: str,
    dotbracket: str,
    output_svg_path: str,
    distro: str | None = None,
    wsl_workdir: str = "/tmp",
    layout_algorithm: int | None = None,
    timeout: int = 60,
) -> str:
    """
    Generate a 2D RNA secondary structure plot using RNAplot (ViennaRNA)
    running inside WSL, and save it as an SVG on the Windows filesystem.

    Parameters
    ----------
    sequence : str
        RNA (or DNA) sequence, e.g. "GGGAAACCC".
    dotbracket : str
        Dot-bracket secondary structure, same length as `sequence`,
        e.g. "(((...)))".
    output_svg_path : str
        Windows path where the SVG should be written, e.g.
        r"C:\\Users\\me\\Desktop\\structure.svg".
    distro : str, optional
        WSL distro name (passed as `wsl -d <distro>`). Uses the default
        distro if not given.
    wsl_workdir : str
        Scratch directory *inside* WSL where RNAplot will briefly write
        its output file before it's streamed back and cleaned up.
        Defaults to "/tmp".
    layout_algorithm : int, optional
        RNAplot's `-t` layout option (0-4). Left as RNAplot's default
        if not given. 1 = naview (default), others are radial/RNApuzzler
        variants depending on RNAplot version.
    timeout : int
        Seconds to wait before giving up on the WSL call.

    Returns
    -------
    str
        The `output_svg_path` that was written.

    Raises
    ------
    ValueError
        If sequence/structure lengths don't match.
    RuntimeError
        If RNAplot exits non-zero or doesn't return valid SVG content.
    """
    if len(sequence) != len(dotbracket):
        raise ValueError(
            f"sequence (len={len(sequence)}) and dotbracket (len={len(dotbracket)}) "
            "must be the same length"
        )

    job_id = "jupyter_rnaplot_job"

    # RNAplot input: an optional FASTA header (used to name output files),
    # then the sequence, then the structure.
    rna_input = f">{job_id}\n{sequence}\n{dotbracket}\n"

    out_filename = f"{job_id}_ss.svg"

    rnaplot_flags = ["-o", "svg"]
    if layout_algorithm is not None:
        rnaplot_flags += ["-t", str(layout_algorithm)]

    # Run inside a login shell in WSL so RNAplot is found on PATH,
    # write to a scratch dir, print the SVG to stdout, then clean up.
    inner_cmd = (
        f"cd {shlex.quote(wsl_workdir)} && "
        f"RNAplot {' '.join(rnaplot_flags)} && "
        f"cat {shlex.quote(out_filename)} && "
        f"rm -f {shlex.quote(out_filename)}"
    )

    wsl_cmd = ["wsl"]
    if distro:
        wsl_cmd += ["-d", distro]
    wsl_cmd += ["--", "bash", "-lc", inner_cmd]

    result = subprocess.run(
        wsl_cmd,
        input=rna_input,     # feeds stdin non-interactively, avoids the hang
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"RNAplot failed (exit code {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    svg_content = result.stdout
    if "<svg" not in svg_content:
        raise RuntimeError(
            "RNAplot did not return valid SVG content. Raw output:\n"
            f"{svg_content[:1000]}\nstderr:\n{result.stderr}"
        )

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    return output_svg_path


if __name__ == "__main__":
    # Example usage
    seq = "GGGAAACCC"
    struct = "(((...)))"
    path = rnaplot_svg(seq, struct, "structure.svg")
    print(f"Saved SVG to {path}")
