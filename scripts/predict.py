"""Load the trained pipeline and write the Kaggle submission.

Command line::

    python -m scripts.predict                      # uses models/pipeline_model.pkl
    python -m scripts.predict --model path/to.pkl --out submissions/submission.csv
"""
import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocess import ID_COL, ROOT, TARGET, load_data, output_dir
from .train import MODEL_NAME


def find_model(path=None):
    """Locate the saved model file.

    Search order: ``path`` argument, ``<output>/models``, ``<repo>/models``
    and any ``pipeline_model.pkl`` under ``/kaggle/input`` (e.g. the
    output of a previous Kaggle notebook added as input).

    Parameters
    ----------
    path : str or Path, optional

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileNotFoundError
        If no model file is found.
    """
    candidates = [Path(path)] if path else []
    candidates += [output_dir() / "models" / MODEL_NAME, ROOT / "models" / MODEL_NAME]
    for cand in candidates:
        if cand.exists():
            return cand
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        hits = sorted(kaggle_input.glob(f"**/{MODEL_NAME}"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{MODEL_NAME} not found - run scripts.train / 02_modeling.ipynb first.")


def load_model(path=None):
    """Load the fitted :class:`~scripts.train.WeightedArgmaxClassifier`.

    Parameters
    ----------
    path : str or Path, optional
        See :func:`find_model`.

    Returns
    -------
    WeightedArgmaxClassifier
    """
    model_path = find_model(path)
    print(f"loading model from {model_path}")
    return joblib.load(model_path)


def predict_grades(model, test):
    """Predict integer damage grades (1..5) for the test table.

    Parameters
    ----------
    model : WeightedArgmaxClassifier
    test : pandas.DataFrame
        Raw ``test.csv`` table.

    Returns
    -------
    numpy.ndarray of int
    """
    return np.asarray(model.predict(test)).astype(int)


def format_submission(sample, ids, grades, text_labels=False):
    """Build the submission with the columns and row order of ``sample_submission.csv``.

    The Kaggle scorer expects **integer** grades 1..5 (an upload with
    ``"Grade 3"`` labels fails with "The input string 'Grade' was not in a
    correct format"), even though ``sample_submission.csv`` shows text labels.

    Parameters
    ----------
    sample : pandas.DataFrame
        ``sample_submission.csv`` (only its ``building_id`` order is used).
    ids : array-like
        ``building_id`` of the predicted rows.
    grades : array-like of int
        Predicted grades 1..5.
    text_labels : bool, default=False
        Write ``"Grade X"`` instead of integers (not accepted by the scorer).

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If some ``building_id`` of the sample has no prediction.
    """
    pred = pd.DataFrame({ID_COL: np.asarray(ids), "_grade": np.asarray(grades).astype(int)})
    sub = sample[[ID_COL]].merge(pred, on=ID_COL, how="left")
    if sub["_grade"].isna().any():
        raise ValueError(f"{int(sub['_grade'].isna().sum())} building_id without prediction")
    grades_int = sub["_grade"].astype(int)
    sub[TARGET] = ("Grade " + grades_int.astype(str)) if text_labels else grades_int
    return sub[[ID_COL, TARGET]]


def save_submission(sub, path=None):
    """Write the submission CSV (``index=False``).

    On Kaggle a copy is also written to ``/kaggle/working/submission.csv``
    so it can be submitted directly from the notebook output.

    Parameters
    ----------
    sub : pandas.DataFrame
    path : str or Path, optional
        Default: ``<output>/submissions/submission.csv``.

    Returns
    -------
    pathlib.Path
    """
    path = Path(path) if path else output_dir() / "submissions" / "submission.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    kaggle_copy = Path("/kaggle/working/submission.csv")
    if kaggle_copy.parent.exists() and kaggle_copy.resolve() != path.resolve():
        shutil.copy(path, kaggle_copy)
    return path


def main(argv=None):
    """Command-line entry point: load model, predict test, save submission.

    Parameters
    ----------
    argv : list of str, optional
    """
    parser = argparse.ArgumentParser(description="Create the Kaggle submission.")
    parser.add_argument("--model", default=None, help="path to the .pkl model")
    parser.add_argument("--out", default=None, help="output CSV path")
    args = parser.parse_args(argv)
    model = load_model(args.model)
    _, test, sample = load_data()
    sub = format_submission(sample, test[ID_COL], predict_grades(model, test))
    path = save_submission(sub, args.out)
    print(f"saved {len(sub):,} rows -> {path}")
    print(sub[TARGET].value_counts(normalize=True).sort_index().round(3).to_string())


if __name__ == "__main__":
    main()
