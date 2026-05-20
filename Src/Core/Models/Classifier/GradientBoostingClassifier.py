"""Gradient Boosting classification pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class GradientBoostingClassifierModel:
    Name = "GradientBoostingClassifier"

    def CreateEstimator(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            random_state=216,
        )

    def CreateParamGrid(self) -> dict[str, list[object]]:
        return {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
            "model__min_samples_leaf": [1, 3],
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
