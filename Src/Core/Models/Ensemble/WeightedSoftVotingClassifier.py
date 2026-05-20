"""Weighted soft voting ensemble for already trained classifiers."""

from typing import Protocol

import numpy


class ProbabilisticEstimator(Protocol):
    classes_: object

    def predict_proba(self, X): ...


class WeightedSoftVotingClassifier:
    """Combine fitted classifiers by averaging their predicted probabilities."""

    Name = "WeightedSoftVotingEnsemble"

    def __init__(
        self,
        estimators: list[tuple[str, ProbabilisticEstimator]],
        weights: list[float],
        predictionThreshold: float = 0.5,
    ):
        if not estimators:
            raise ValueError("At least one estimator is required.")
        if len(estimators) != len(weights):
            raise ValueError("Estimators and weights must have the same length.")
        if predictionThreshold <= 0 or predictionThreshold >= 1:
            raise ValueError("predictionThreshold must be greater than 0 and less than 1.")

        self.estimators = estimators
        self.weights = weights
        self.predictionThreshold = predictionThreshold
        self.classes_ = getattr(estimators[0][1], "classes_", numpy.array([False, True]))

    def predict_proba(self, X):
        probabilities = []

        for _, estimator in self.estimators:
            if not hasattr(estimator, "predict_proba"):
                raise ValueError("All estimators must support predict_proba.")

            probabilities.append(estimator.predict_proba(X))

        return numpy.average(probabilities, axis=0, weights=self.weights)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        classes = list(self.classes_)

        if True in classes:
            positiveClassIndex = classes.index(True)
        else:
            positiveClassIndex = len(classes) - 1

        negativeClassIndex = 1 - positiveClassIndex
        class_indexes = numpy.where(
            probabilities[:, positiveClassIndex] >= self.predictionThreshold,
            positiveClassIndex,
            negativeClassIndex,
        )

        return self.classes_[class_indexes]

    def get_metadata(self) -> dict[str, object]:
        return {
            "EstimatorNames": [name for name, _ in self.estimators],
            "Weights": self.weights,
            "PredictionThreshold": self.predictionThreshold,
        }
