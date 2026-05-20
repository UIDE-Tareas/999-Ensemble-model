"""Histogram Gradient Boosting classification pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class HistGradientBoostingClassifierModel:
    Name = "HistGradientBoostingClassifier"

    def CreateEstimator(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            random_state=216,
            early_stopping=True,
        )

    def CreateParamGrid(self) -> dict[str, list[object]]:
        return {
            "model__learning_rate": [0.03, 0.1],
            "model__max_iter": [100, 200],
            "model__max_leaf_nodes": [15, 31],
            "model__l2_regularization": [0.0, 0.1],
            "model__min_samples_leaf": [10, 20],
        }

    def CreatePipeline(
        self,
        categoricalFeatures: list[str],
        numericFeatures: list[str],
    ) -> Pipeline:
        categoricalTransformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        numericTransformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categoricalTransformer, categoricalFeatures),
                ("numeric", numericTransformer, numericFeatures),
            ],
            sparse_threshold=0.0,
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", self.CreateEstimator()),
            ]
        )
