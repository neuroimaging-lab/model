from itertools import product
import subprocess

from segmentation.config import CLUSTER_TRAINING_ENABLED


def fine_tune():
    gamma: list[float] = [2.0, 3.0, 4.0, 5.0]
    alpha: list[float] = [0.5, 0.75, 0.9, 0.99]
    param_combinations: list[tuple] = list(product(gamma, alpha))

    succes_iterations: int = 0
    for i, params in enumerate(param_combinations):
        y, a = params
        command = [
            "uv",
            "run",
            "-m",
            "segmentation.examples.train",
            "--gamma",
            str(y),
            "--alpha",
            str(a),
        ]

        if CLUSTER_TRAINING_ENABLED:
            command.insert(2, "--directory")
            command.insert(3, "model")

        print("\n" + "-" * 40)
        print(f"Running {i + 1}/{len(param_combinations)} test...")
        print("-" * 40 + "\n")
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Process failed with parameters: {params}")
            break
        succes_iterations += 1
        if succes_iterations == 2:
            return

    print(
        f"End of fine tunning! {succes_iterations}/{len(param_combinations)} success iterations!"
    )


if __name__ == "__main__":
    fine_tune()
