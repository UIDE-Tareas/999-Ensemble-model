"""Support Vector Machine classification pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class SupportVectorClassifierModel:
    Name = "SupportVectorClassifier"

    def CreateEstimator(self) -> SVC:
        return SVC(
            probability=True,
            random_state=216,
        )

    def CreateParamGrid(self) -> list[dict[str, list[object]]]:
        return [
            {
                "model__kernel": ["linear"],
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
            {
                "model__kernel": ["rbf"],
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", "auto"],
                "model__class_weight": [None, "balanced"],
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
