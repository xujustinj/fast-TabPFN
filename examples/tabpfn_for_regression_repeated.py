#  Copyright (c) Prior Labs GmbH 2025.
"""Example of using TabPFN for regression.

This example demonstrates how to use TabPFNRegressor on a regression task
using the diabetes dataset from scikit-learn.
"""

import argparse
from pathlib import Path
from typing import Union

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor
from tabpfn.profiling import push_tags, timer, timing_summary

X, y = load_diabetes(return_X_y=True)


def main():
    reg = TabPFNRegressor()

    for i in range(5):
        yield "=" * 80
        yield f"FIT {i=}"

        with push_tags("i=0" if i == 0 else "i>0"):
            # Load data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.33,
                random_state=i,
            )

            # Initialize a regressor
            with timer("fit"):
                reg.fit(X_train, y_train)

            # Predict a point estimate (using the mean)
            with timer("predict_mean"):
                predictions = reg.predict(X_test)

            yield (
                f"Mean Squared Error (MSE): {mean_squared_error(y_test, predictions)}"
            )
            yield (
                f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions)}"
            )
            yield f"R-squared (R^2): {r2_score(y_test, predictions)}"

            # Predict quantiles
            with timer("predict_quantiles"):
                quantiles = [0.25, 0.5, 0.75]
                quantile_predictions = reg.predict(
                    X_test,
                    output_type="quantiles",
                    quantiles=quantiles,
                )
            for q, q_pred in zip(quantiles, quantile_predictions):
                yield f"Quantile {q} MAE: {mean_absolute_error(y_test, q_pred)}"

            # Predict with mode
            with timer("predict_mode"):
                mode_predictions = reg.predict(X_test, output_type="mode")
            yield f"Mode MAE: {mean_absolute_error(y_test, mode_predictions)}"

        yield ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    base_path = Path("examples/output")
    if args.name:
        base_path = base_path / args.name
    base_path.mkdir(parents=True, exist_ok=True)

    output: Union[list[str], None] = None
    assert args.trials > 0
    for _ in range(args.trials):
        trial_output = list(main())
        if output is None:
            output = trial_output
        else:
            assert len(output) == len(trial_output)
    assert output is not None

    with (base_path / "tabpfn_for_regression_repeated.txt").open("w") as f:
        for line in output:
            f.write(line + "\n")

        f.write("=" * 80 + "\n")
        f.write("TIMING SUMMARY" + "\n")
        for line in timing_summary():
            f.write(line + "\n")
