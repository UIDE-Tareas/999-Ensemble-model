"""Logistic Regression classification pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:
    Name = "LogisticRegression"

    def CreateEstimator(self) -> LogisticRegression:
        return LogisticRegression(
            max_iter=2000,
            random_state=216,
        )

    def CreateParamGrid(self) -> list[dict[str, list[object]]]:
        return [
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["liblinear"],
            },
        ]

    def CreatePipeline(
        self,
        categoricalFeatures: list[str],
        numericFeatures: list[str],
    ) -> Pipeline:
        categoricalTransformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        numericTransformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categoricalTransformer, categoricalFeatures),
                ("numeric", numericTransformer, numericFeatures),
            ]
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", self.CreateEstimator()),
            ]
        )
