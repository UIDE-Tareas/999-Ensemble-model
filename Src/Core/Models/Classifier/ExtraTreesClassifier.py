"""Extra Trees classification pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class ExtraTreesClassifierModel:
    Name = "ExtraTreesClassifier"

    def CreateEstimator(self) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=300,
            random_state=216,
            n_jobs=1,
            class_weight="balanced",
        )

    def CreateParamGrid(self) -> dict[str, list[object]]:
        return {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": [None, "balanced"],
        }

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
