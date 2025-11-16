from itertools import product
import subprocess


def fine_tune():
    gamma: list[float] = [2.0]
    alpha: list[float] = [0.75]
    learning_rate: list[float] = [5e-5]
    dropout_rate: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    param_combinations: list[tuple] = list(
        product(gamma, alpha, learning_rate, dropout_rate)
    )

    success_iterations: int = 0
    for i, params in enumerate(param_combinations):
        y, a, lr, dr = params
        command = [
            "uv",
            "run",
            "-m",
            "segmentation.examples.train",
            "--gamma",
            str(y),
            "--alpha",
            str(a),
            "--lr",
            str(lr),
            "--dropout-rate",
            str(dr),
            "--store-checkpoints-in-temp-storage",
        ]

        print("\n" + "-" * 40, flush=True)
        print(f"Running {i + 1}/{len(param_combinations)} test...", flush=True)
        print("-" * 40 + "\n", flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Process failed with parameters: {params}")
            continue
        success_iterations += 1

    print(
        f"End of fine tuning! {success_iterations}/{len(param_combinations)} success iterations!"
    )


if __name__ == "__main__":
    fine_tune()
