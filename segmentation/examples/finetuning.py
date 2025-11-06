from itertools import product
import subprocess


def fine_tune():
    # gamma: list[float] = [2.0, 3.0, 4.0, 5.0]
    # alpha: list[float] = [0.5, 0.75, 0.9, 0.99]
    gamma: list[float] = [2.0]
    alpha: list[float] = [0.5]
    learning_rate: list[float] = [5e-5, 1e-4, 2e-4, 3e-4, 1e-3]
    
    param_combinations: list[tuple] = list(product(gamma, alpha, learning_rate))
   
    succes_iterations: int = 0
    for i, params in enumerate(param_combinations):
        y, a, lr = params
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
            "--store-checkpoints-in-temp-storage",
        ]

        print("\n" + "-" * 40, flush=True)
        print(f"Running {i + 1}/{len(param_combinations)} test...", flush=True)
        print("-" * 40 + "\n", flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Process failed with parameters: {params}")
            continue
        succes_iterations += 1

    print(
        f"End of fine tunning! {succes_iterations}/{len(param_combinations)} success iterations!"
    )


if __name__ == "__main__":
    fine_tune()
