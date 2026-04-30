import subprocess

KINFOLD = "/Users/weberlin/miniconda3/bin/Kinfold"


def run_kinfold(sequence, num=10, time=100, temp=37, seed=None, extra_args=None):

    args = [KINFOLD, "--num", str(num), "--time", str(time), "--Temp", str(temp)]
    if seed is not None:
        args += ["--seed", f"{seed}={seed}={seed}"]
    if extra_args:
        args += extra_args

    stdin_input = f"{sequence}"

    result = subprocess.run(
        args,
        input=stdin_input,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Kinfold failed:\n{result.stderr}")

    fpt_times = []

    timed_out = 0

    for line in result.stdout.splitlines():
        if "X1" in line:
            parts = line.split()
            fpt_times.append(float(parts[2]))  # time is the 3rd column

    timed_out = num - len(fpt_times)

    return {
        "fpt_times": fpt_times,
        "reached_mfe": len(fpt_times),
        "timed_out": timed_out,
    }

# Example usage
if __name__ == "__main__":
    seq = "GGGAAACCAUGCUAGCUAGACUCAUCGAUGCGCGGGAAACCAUGCUAGCUAGACUCAUCGAUGCGC"
    result = run_kinfold(seq, num=5, time=10000000000, seed=229)
    print(f"FPT times: {result['fpt_times']}")
    print(f"Reached MFE: {result['reached_mfe']}/5")
    print(f"Timed out:   {result['timed_out']}/5")