"""Training entry point for the predictive maintenance ensemble model."""

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol
from typing import cast
from typing import TypedDict

import joblib
import pandas
from Core.Models.Ensemble.WeightedSoftVotingClassifier import ProbabilisticEstimator
from Core.Models.Ensemble.WeightedSoftVotingClassifier import WeightedSoftVotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit

from Core.Dataframes import Dataset
from Core.Dataframes import DatasetSplit
from Core.Dataframes import DropColumns
from Core.Dataframes import NormalizeColumnNames
from Core.Dataframes import PrintData
from Core.Dataframes import PrintShowSeparator
from Core.Dataframes import RemoveDfDuplicates
from Core.Dataframes import RemoveDfNanValues
from Core.Dataframes import ShowDfDataTypes
from Core.Dataframes import ShowDfDateRanges
from Core.Dataframes import ShowDfDuplicates
from Core.Dataframes import ShowDfHead
from Core.Dataframes import ShowDfInfo
from Core.Dataframes import ShowDfMemoryUsage
from Core.Dataframes import ShowDfNanRows
from Core.Dataframes import ShowDfShape
from Core.Dataframes import ShowDfStats
from Core.Dataframes import ShowDfTail
from Core.Dataframes import ShowDfUniqueValues
from Core.Dataframes import ShowDatasetSplitHead
from Core.Models.Classifier.AdaBoostClassifier import AdaBoostClassifierModel
from Core.Models.Classifier.DecisionTreeClassifier import DecisionTreeClassifierModel
from Core.Models.Classifier.ExtraTreesClassifier import ExtraTreesClassifierModel
from Core.Models.Classifier.GaussianNaiveBayes import GaussianNaiveBayesModel
from Core.Models.Classifier.GradientBoostingClassifier import GradientBoostingClassifierModel
from Core.Models.Classifier.HistGradientBoostingClassifier import HistGradientBoostingClassifierModel
from Core.Models.Classifier.KNeighborsClassifier import KNeighborsClassifierModel
from Core.Models.Classifier.LogisticRegression import LogisticRegressionModel
from Core.Models.Classifier.RandomForestClassifier import RandomForestClassifierModel
from Core.Models.Classifier.SupportVectorClassifier import SupportVectorClassifierModel
from Core.Utils import ConsoleColor
from Core.Utils import GetVisibleLength
from Core.Utils import PrintSuccess
from Core.Utils import PrintWarning
from Core.Utils import ShowEnvironmentInfo
from Core.Utils import ShowTitleBox
from Core.Utils import ShowTitleBoxWithSpacing

TRAIN_BANNER = r"""
╭─╴╭╮╷╭─╮╭─╴╭┬╮╭╮ ╷  ╭─╴ ╭┬╮╭─╮╶┬╮╭─╴╷   ╶┬╴╭─╮╭─╮╷╭╮╷
├╴ │╰┤╰─╮├╴ │││├┴╮│  ├╴  ││││ │ ││├╴ │    │ ├┬╯├─┤││╰┤
╰─╴╵ ╵╰─╯╰─╴╵ ╵╰─╯╰─╴╰─╴ ╵ ╵╰─╯╶┴╯╰─╴╰─╴  ╵ ╵╰╴╵ ╵╵╵ ╵
"""

GridSearchParamGrid = dict[str, list[object]] | list[dict[str, list[object]]]


class TrainingColumnSet(TypedDict):
    TargetColumn: str
    ExcludedColumns: list[str]
    FeatureColumns: list[str]


class FeatureColumnTypes(TypedDict):
    CategoricalFeatures: list[str]
    NumericFeatures: list[str]


class ClassificationModelDefinition(Protocol):
    Name: str

    def CreateParamGrid(self) -> GridSearchParamGrid: ...

    def CreatePipeline(
        self,
        categoricalFeatures: list[str],
        numericFeatures: list[str],
    ) -> Pipeline: ...


ModelFactory = type[ClassificationModelDefinition]


class ModelPipelineInfo(TypedDict):
    Name: str
    Pipeline: Pipeline
    ParamGrid: GridSearchParamGrid


class TrainedModelInfo(TypedDict):
    TargetName: str
    ModelName: str
    BestEstimator: object
    BestParams: dict[str, object]
    BestCVScore: float
    BestCVSplits: int
    PredictionThreshold: float
    TestAccuracy: float
    TestPrecision: float
    TestRecall: float
    TestF1: float
    TestRocAuc: float


class TrainingRunResult(TypedDict):
    StartedAt: str
    EndedAt: str
    ElapsedSeconds: float
    Scoring: str
    CvSplitOptions: list[int]
    PredictionThreshold: float
    BestModels: list[TrainedModelInfo]
    AllResults: list[dict[str, object]]
    WinningResults: list[dict[str, object]]


AVAILABLE_MODEL_FACTORIES: dict[str, ModelFactory] = {
    LogisticRegressionModel.Name: LogisticRegressionModel,
    RandomForestClassifierModel.Name: RandomForestClassifierModel,
    DecisionTreeClassifierModel.Name: DecisionTreeClassifierModel,
    ExtraTreesClassifierModel.Name: ExtraTreesClassifierModel,
    GradientBoostingClassifierModel.Name: GradientBoostingClassifierModel,
    HistGradientBoostingClassifierModel.Name: HistGradientBoostingClassifierModel,
    AdaBoostClassifierModel.Name: AdaBoostClassifierModel,
    SupportVectorClassifierModel.Name: SupportVectorClassifierModel,
    KNeighborsClassifierModel.Name: KNeighborsClassifierModel,
    GaussianNaiveBayesModel.Name: GaussianNaiveBayesModel,
}


def RemoveDuplicatedModelNames(modelNames: list[str]) -> list[str]:
    """Remove duplicated model names while preserving the original order."""
    uniqueModelNames = []
    seenModelNames = set()

    for modelName in modelNames:
        if modelName in seenModelNames:
            continue

        uniqueModelNames.append(modelName)
        seenModelNames.add(modelName)

    return uniqueModelNames


def GetDefaultTargetNames() -> list[str]:
    """Get the configured target names for a quick training run."""
    return [
        # "Falla7Dias",
        # "Falla15Dias",
        "Falla30Dias",
    ]


def GetDefaultEnabledModelNames() -> list[str]:
    """Get the configured model names for a quick training run."""
    return [
        LogisticRegressionModel.Name,
        GaussianNaiveBayesModel.Name,
        RandomForestClassifierModel.Name,
        GradientBoostingClassifierModel.Name,
        DecisionTreeClassifierModel.Name,
        ExtraTreesClassifierModel.Name,
        HistGradientBoostingClassifierModel.Name,
        AdaBoostClassifierModel.Name,
        SupportVectorClassifierModel.Name,
        KNeighborsClassifierModel.Name,
    ]


def GetDefaultCvSplitOptions() -> list[int]:
    """Get the configured cross validation split options for a quick training run."""
    return [
        5,
        # 3,
        # 7,
        # 9,
    ]


def GetDefaultPredictionThreshold() -> float:
    """Get the configured probability threshold used to convert probabilities into classes."""
    return 0.5


def CenterMultilineText(text: str, max_width: int = 120) -> str:
    lines = text.strip("\n").splitlines()
    content_width = max(max_width, max(GetVisibleLength(line) for line in lines))
    centered_lines = []

    for line in lines:
        left_padding = max((content_width - GetVisibleLength(line)) // 2, 0)
        centered_lines.append(f"{' ' * left_padding}{line}")

    return "\n".join(centered_lines)


def ShowTrainBanner():
    try:
        from rich.align import Align
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        banner_width = 120
        panel_horizontal_padding = 4
        panel_border_width = 2
        panel_content_width = banner_width - panel_horizontal_padding - panel_border_width
        centered_banner = CenterMultilineText(
            TRAIN_BANNER,
            max_width=panel_content_width,
        )
        panel = Panel(
            Text(centered_banner, style="bold cyan"),
            border_style="cyan",
            padding=(1, 2),
            width=banner_width,
        )
        console.print(
            Align.center(panel)
        )
    except Exception:
        print(CenterMultilineText(TRAIN_BANNER))


def LoadAndInspectDataset():
    """Load the Excel dataset and show a full DataFrame overview."""
    ShowTitleBox(
        "📊 Dataset Inspection",
        max_len=120,
        color=ConsoleColor.MAGENTA,
    )

    dataset_path = Path("Data") / "PRE_DATASET.xlsx"
    df_raw = pandas.read_excel(dataset_path)
    dataset_title = "PRE_DATASET.xlsx"

    ShowDfInfo(df_raw, dataset_title)
    ShowDfShape(df_raw, dataset_title)
    ShowDfDataTypes(df_raw, dataset_title)
    ShowDfMemoryUsage(df_raw, dataset_title)
    ShowDfDateRanges(df_raw, dataset_title)
    ShowDfStats(df_raw, dataset_title)
    ShowDfDuplicates(df_raw, dataset_title)
    ShowDfNanRows(df_raw, dataset_title)
    ShowDfUniqueValues(df_raw, dataset_title)
    ShowDfHead(df_raw, dataset_title, headQty=10)
    ShowDfTail(df_raw, dataset_title, tailQty=10)

    ShowTitleBox(
        "🧹 Dataset Cleaning And Transformations",
        max_len=120,
        color=ConsoleColor.YELLOW,
    )

    df_clean = NormalizeColumnNames(df_raw.copy())
    PrintSuccess("Column names were normalized.", "Columns Normalized")

    df_clean = DropColumns(df_clean, ["Elemento"])
    PrintWarning(
        'Column "Elemento" was removed because "ElementoEstandarizado" contains the standardized replacement.',
        "Feature Removed",
    )
    rows_before_nan_cleanup = df_clean.shape[0]
    df_clean = RemoveDfNanValues(df_clean)
    removed_nan_rows = rows_before_nan_cleanup - df_clean.shape[0]
    PrintWarning(
        f"{removed_nan_rows} rows with null values were removed.",
        "Null Rows Removed",
    )
    rows_before_duplicate_cleanup = df_clean.shape[0]
    df_clean = RemoveDfDuplicates(df_clean).reset_index(drop=True)
    removed_duplicate_rows = rows_before_duplicate_cleanup - df_clean.shape[0]
    PrintWarning(
        f"{removed_duplicate_rows} duplicate rows were removed.",
        "Duplicate Rows Removed",
    )

    df_clean = df_clean.sort_values(by="FechaYHoraDeFalla").reset_index(drop=True)
    PrintSuccess(
        'Rows were sorted by "FechaYHoraDeFalla".',
        "Rows Sorted",
    )

    PrintShowSeparator()
    ShowDfInfo(df_clean, "PRE_DATASET.xlsx Clean Dataset")
    ShowDfHead(df_clean, "PRE_DATASET.xlsx Clean Dataset", headQty=10)

    return df_clean


def BuildDailyFailuresDataFrame(df_clean: pandas.DataFrame) -> pandas.DataFrame:
    """Build the daily failure aggregation for this dataset."""
    group_columns = [
        "Maquina",
        "Producto",
        "TipoDeFalla",
        "ElementoEstandarizado",
    ]

    df_daily_source = df_clean.copy()
    df_daily_source["FechaYHoraDeFalla"] = pandas.to_datetime(
        df_daily_source["FechaYHoraDeFalla"]
    )
    df_daily_source["FechaDeFalla"] = df_daily_source["FechaYHoraDeFalla"].dt.date

    df_daily_failures = (
        df_daily_source.groupby(["FechaDeFalla", *group_columns], as_index=False)
        .agg(
            TiempoTotalDeFallaMin=("TiempoTotalDeFallaMin", "sum"),
            CantidadTotalFallas=("TiempoTotalDeFallaMin", "count"),
        )
        .sort_values(["FechaDeFalla", *group_columns])
        .reset_index(drop=True)
    )

    df_daily_failures["FechaDeFalla"] = pandas.to_datetime(
        df_daily_failures["FechaDeFalla"]
    )
    df_daily_failures["DiaSemana"] = df_daily_failures["FechaDeFalla"].dt.dayofweek
    df_daily_failures["DiaMes"] = df_daily_failures["FechaDeFalla"].dt.day
    df_daily_failures["Mes"] = df_daily_failures["FechaDeFalla"].dt.month
    df_daily_failures["EsFinSemana"] = df_daily_failures["DiaSemana"].isin([5, 6])

    ordered_columns = [
        "FechaDeFalla",
        *group_columns,
        "TiempoTotalDeFallaMin",
        "CantidadTotalFallas",
        "DiaSemana",
        "DiaMes",
        "Mes",
        "EsFinSemana",
    ]
    df_daily_failures = df_daily_failures[ordered_columns]

    ShowDfInfo(df_daily_failures, "df_daily_failures")
    ShowDfUniqueValues(df_daily_failures, "df_daily_failures")
    ShowDfHead(df_daily_failures, "df_daily_failures", headQty=10)

    return df_daily_failures


def BuildMachineDailySeriesDataFrame(
    df_clean: pandas.DataFrame,
    df_daily_failures: pandas.DataFrame,
) -> pandas.DataFrame:
    """Build a daily machine series with non-failure days and previous known context."""
    machines = sorted(df_clean["Maquina"].dropna().unique())
    start_date = df_clean["FechaYHoraDeFalla"].min().normalize()
    end_date = df_clean["FechaYHoraDeFalla"].max().normalize()
    date_range = pandas.date_range(start=start_date, end=end_date, freq="D")

    df_machine_daily_series = (
        pandas.MultiIndex.from_product(
            [date_range, machines],
            names=["Fecha", "Maquina"],
        )
        .to_frame(index=False)
        .sort_values(["Maquina", "Fecha"])
        .reset_index(drop=True)
    )

    df_machine_daily_failures = (
        df_daily_failures.groupby(["FechaDeFalla", "Maquina"], as_index=False)
        .agg(
            TiempoTotalDeFallaMin=("TiempoTotalDeFallaMin", "sum"),
            CantidadTotalFallas=("CantidadTotalFallas", "sum"),
        )
        .rename(columns={"FechaDeFalla": "Fecha"})
    )

    df_machine_daily_series = df_machine_daily_series.merge(
        df_machine_daily_failures,
        on=["Fecha", "Maquina"],
        how="left",
    )
    df_machine_daily_series["TiempoTotalDeFallaMin"] = (
        df_machine_daily_series["TiempoTotalDeFallaMin"].fillna(0)
    )
    df_machine_daily_series["CantidadTotalFallas"] = (
        df_machine_daily_series["CantidadTotalFallas"].fillna(0).astype(int)
    )
    df_machine_daily_series["FalloDia"] = (
        df_machine_daily_series["CantidadTotalFallas"] > 0
    )

    df_event_context = df_clean[
        [
            "FechaYHoraDeFalla",
            "Maquina",
            "Producto",
            "TipoDeFalla",
            "ElementoEstandarizado",
        ]
    ].copy()
    df_event_context["Fecha"] = (
        pandas.to_datetime(df_event_context["FechaYHoraDeFalla"]).dt.normalize()
        + pandas.Timedelta(days=1)
    )
    df_event_context = (
        df_event_context.sort_values(["Maquina", "FechaYHoraDeFalla"])
        .groupby(["Maquina", "Fecha"], as_index=False)
        .tail(1)
        .sort_values(["Maquina", "Fecha"])
    )
    df_event_context = df_event_context[
        [
            "Fecha",
            "Maquina",
            "Producto",
            "TipoDeFalla",
            "ElementoEstandarizado",
        ]
    ].rename(
        columns={
            "Producto": "UltimoProductoConocido",
            "TipoDeFalla": "UltimoTipoFallaConocido",
            "ElementoEstandarizado": "UltimoElementoEstandarizadoConocido",
        }
    )

    df_machine_daily_series = df_machine_daily_series.merge(
        df_event_context,
        on=["Fecha", "Maquina"],
        how="left",
    )
    context_columns = [
        "UltimoProductoConocido",
        "UltimoTipoFallaConocido",
        "UltimoElementoEstandarizadoConocido",
    ]
    df_machine_daily_series[context_columns] = (
        df_machine_daily_series.groupby("Maquina")[context_columns].ffill()
    )
    df_machine_daily_series[context_columns] = df_machine_daily_series[
        context_columns
    ].fillna("SIN_HISTORIAL")

    df_machine_daily_series["DiaSemana"] = df_machine_daily_series["Fecha"].dt.dayofweek
    df_machine_daily_series["DiaMes"] = df_machine_daily_series["Fecha"].dt.day
    df_machine_daily_series["Mes"] = df_machine_daily_series["Fecha"].dt.month
    df_machine_daily_series["EsFinSemana"] = df_machine_daily_series["DiaSemana"].isin(
        [5, 6]
    )

    ordered_columns = [
        "Fecha",
        "Maquina",
        "FalloDia",
        "CantidadTotalFallas",
        "TiempoTotalDeFallaMin",
        "UltimoProductoConocido",
        "UltimoTipoFallaConocido",
        "UltimoElementoEstandarizadoConocido",
        "DiaSemana",
        "DiaMes",
        "Mes",
        "EsFinSemana",
    ]
    df_machine_daily_series = df_machine_daily_series[ordered_columns]

    ShowDfInfo(df_machine_daily_series, "df_machine_daily_series")
    ShowDfHead(df_machine_daily_series, "df_machine_daily_series", headQty=10)

    return df_machine_daily_series


def AddHistoricalFailureFeatures(
    df_machine_daily_series: pandas.DataFrame,
    df_clean: pandas.DataFrame,
    windows: list[int] | None = None,
) -> pandas.DataFrame:
    """Add historical failure features for each machine and date."""
    if windows is None:
        windows = [7, 15, 30]

    df_features = df_machine_daily_series.copy()
    df_events = df_clean[
        [
            "FechaYHoraDeFalla",
            "Maquina",
            "TiempoTotalDeFallaMin",
            "TipoDeFalla",
            "ElementoEstandarizado",
            "Producto",
        ]
    ].copy()
    df_events["FechaDeFalla"] = pandas.to_datetime(
        df_events["FechaYHoraDeFalla"]
    ).dt.normalize()

    feature_rows = []

    for row in df_features[["Fecha", "Maquina"]].to_dict("records"):
        current_date = pandas.Timestamp(row["Fecha"])
        machine = str(row["Maquina"])
        feature_row = {
            "Fecha": current_date,
            "Maquina": machine,
        }
        df_machine_events = df_events[df_events["Maquina"] == machine]

        for window in windows:
            start_date = current_date - pandas.Timedelta(days=window)
            end_date = current_date - pandas.Timedelta(days=1)
            df_window = df_machine_events[
                (df_machine_events["FechaDeFalla"] >= start_date)
                & (df_machine_events["FechaDeFalla"] <= end_date)
            ]

            suffix = f"Ultimos{window}Dias"
            feature_row[f"Fallas{suffix}"] = int(df_window.shape[0])
            feature_row[f"TiempoFalla{suffix}"] = float(
                df_window["TiempoTotalDeFallaMin"].sum()
            )
            feature_row[f"CantidadTiposFallaDistintos{suffix}"] = int(
                df_window["TipoDeFalla"].nunique()
            )
            feature_row[f"CantidadElementosDistintos{suffix}"] = int(
                df_window["ElementoEstandarizado"].nunique()
            )
            feature_row[f"CantidadProductosDistintos{suffix}"] = int(
                df_window["Producto"].nunique()
            )

            if df_window.empty:
                feature_row[f"TipoFallaMasFrecuente{suffix}"] = "SIN_HISTORIAL"
                feature_row[f"ElementoMasFrecuente{suffix}"] = "SIN_HISTORIAL"
                feature_row[f"ProductoMasFrecuente{suffix}"] = "SIN_HISTORIAL"
            else:
                feature_row[f"TipoFallaMasFrecuente{suffix}"] = (
                    df_window["TipoDeFalla"].mode().iloc[0]
                )
                feature_row[f"ElementoMasFrecuente{suffix}"] = (
                    df_window["ElementoEstandarizado"].mode().iloc[0]
                )
                feature_row[f"ProductoMasFrecuente{suffix}"] = (
                    df_window["Producto"].mode().iloc[0]
                )

        feature_rows.append(feature_row)

    df_historical_features = pandas.DataFrame(feature_rows)
    df_features = df_features.merge(
        df_historical_features,
        on=["Fecha", "Maquina"],
        how="left",
    )

    ShowDfInfo(df_features, "df_machine_daily_series_with_history")
    ShowDfHead(df_features, "df_machine_daily_series_with_history", headQty=10)

    return df_features


def AddFailureTargetColumns(
    df_machine_daily_series: pandas.DataFrame,
    windows: list[int] | None = None,
) -> pandas.DataFrame:
    """Add future failure target columns for each machine."""
    if windows is None:
        windows = [7, 15, 30]

    df_targets = df_machine_daily_series.copy()
    df_targets = df_targets.sort_values(["Maquina", "Fecha"]).reset_index(drop=True)

    for window in windows:
        target_column = f"FallaProximos{window}Dias"
        df_targets[target_column] = False

        for machine, machine_index in df_targets.groupby("Maquina").groups.items():
            machine_failure_series = df_targets.loc[machine_index, "FalloDia"]
            future_failure_count = (
                machine_failure_series.shift(-1)
                .rolling(window=window, min_periods=1)
                .sum()
                .shift(-(window - 1))
            )
            df_targets.loc[machine_index, target_column] = (
                future_failure_count.fillna(0) > 0
            ).astype(bool)

    ShowDfInfo(df_targets, "df_machine_daily_series_with_targets")
    ShowDfHead(df_targets, "df_machine_daily_series_with_targets", headQty=10)

    return df_targets


def BuildFeatureTargetColumnSets(
    df_training_dataset: pandas.DataFrame,
) -> dict[str, TrainingColumnSet]:
    """Define feature columns, target columns, and excluded columns for training."""
    targetFalla7Dias = "FallaProximos7Dias"
    targetFalla15Dias = "FallaProximos15Dias"
    targetFalla30ProximosDias = "FallaProximos30Dias"

    baseExcludedColumns = [
        "Fecha",
        "FalloDia",
        "CantidadTotalFallas",
        "TiempoTotalDeFallaMin",
    ]
    targetColumns = [
        targetFalla7Dias,
        targetFalla15Dias,
        targetFalla30ProximosDias,
    ]

    commonExcludedColumns = [
        *baseExcludedColumns,
        *targetColumns,
    ]

    candidateFeatureColumns = [
        column
        for column in df_training_dataset.columns
        if column not in [*baseExcludedColumns, *targetColumns]
    ]
    historicalWindowSuffixes = [
        "Ultimos7Dias",
        "Ultimos15Dias",
        "Ultimos30Dias",
    ]
    historicalFeatureColumns = [
        column
        for column in candidateFeatureColumns
        if any(
            column.endswith(historicalWindowSuffix)
            for historicalWindowSuffix in historicalWindowSuffixes
        )
    ]
    nonHistoricalFeatureColumns = [
        column
        for column in candidateFeatureColumns
        if column not in historicalFeatureColumns
    ]

    def buildFeatureColumns(allowedHistoricalWindowSuffixes: list[str]) -> list[str]:
        allowedHistoricalFeatureColumns = [
            column
            for column in historicalFeatureColumns
            if any(
                column.endswith(allowedHistoricalWindowSuffix)
                for allowedHistoricalWindowSuffix in allowedHistoricalWindowSuffixes
            )
        ]

        return [
            column
            for column in candidateFeatureColumns
            if column in nonHistoricalFeatureColumns
            or column in allowedHistoricalFeatureColumns
        ]

    featureColumnsFalla7Dias = buildFeatureColumns(["Ultimos7Dias"])
    featureColumnsFalla15Dias = buildFeatureColumns(
        ["Ultimos7Dias", "Ultimos15Dias"]
    )
    featureColumnsFalla30Dias = buildFeatureColumns(
        ["Ultimos7Dias", "Ultimos15Dias", "Ultimos30Dias"]
    )

    column_sets: dict[str, TrainingColumnSet] = {
        "Falla7Dias": {
            "TargetColumn": targetFalla7Dias,
            "ExcludedColumns": [
                *commonExcludedColumns,
                *[
                    column
                    for column in historicalFeatureColumns
                    if column not in featureColumnsFalla7Dias
                ],
            ],
            "FeatureColumns": featureColumnsFalla7Dias,
        },
        "Falla15Dias": {
            "TargetColumn": targetFalla15Dias,
            "ExcludedColumns": [
                *commonExcludedColumns,
                *[
                    column
                    for column in historicalFeatureColumns
                    if column not in featureColumnsFalla15Dias
                ],
            ],
            "FeatureColumns": featureColumnsFalla15Dias,
        },
        "Falla30Dias": {
            "TargetColumn": targetFalla30ProximosDias,
            "ExcludedColumns": commonExcludedColumns,
            "FeatureColumns": featureColumnsFalla30Dias,
        },
    }

    summary_rows = []
    for set_name, config in column_sets.items():
        summary_rows.append(
            {
                "SetName": set_name,
                "TargetColumn": config["TargetColumn"],
                "FeatureColumnsCount": len(config["FeatureColumns"]),
                "ExcludedColumnsCount": len(config["ExcludedColumns"]),
                "FeatureColumns": ", ".join(config["FeatureColumns"]),
                "ExcludedColumns": ", ".join(config["ExcludedColumns"]),
            }
        )

    PrintSuccess("Feature and target column sets were created.", "Column Sets")
    PrintData(pandas.DataFrame(summary_rows))
    PrintShowSeparator()

    return column_sets


def ValidateTargetNames(
    targetNames: list[str],
    trainingColumnSets: dict[str, TrainingColumnSet],
) -> list[str]:
    """Validate configured target names and remove duplicates preserving order."""
    uniqueTargetNames = []
    seenTargetNames = set()

    for targetName in targetNames:
        if targetName in seenTargetNames:
            continue

        uniqueTargetNames.append(targetName)
        seenTargetNames.add(targetName)

    unknownTargetNames = [
        targetName
        for targetName in uniqueTargetNames
        if targetName not in trainingColumnSets
    ]
    if unknownTargetNames:
        availableTargetNames = ", ".join(trainingColumnSets.keys())
        raise ValueError(
            f"Unknown target names: {unknownTargetNames}. Available targets: {availableTargetNames}"
        )

    return uniqueTargetNames


def GetSplitGapDaysByTargetName(targetName: str) -> int:
    """Get the temporal split gap in days for each target horizon."""
    splitGapDaysByTargetName = {
        "Falla7Dias": 7,
        "Falla15Dias": 15,
        "Falla30Dias": 30,
    }

    return splitGapDaysByTargetName[targetName]


def BuildTemporalTrainTestSplit(
    df_training_dataset: pandas.DataFrame,
    featureColumns: list[str],
    targetColumn: str,
    dateColumn: str = "Fecha",
    trainRatio: float = 0.8,
    splitGapDays: int = 0,
) -> DatasetSplit:
    """Build a temporal train/test split using older dates for train and newer dates for test."""
    if dateColumn not in df_training_dataset.columns:
        raise ValueError(f'Date column "{dateColumn}" does not exist.')
    if targetColumn not in df_training_dataset.columns:
        raise ValueError(f'Target column "{targetColumn}" does not exist.')

    missing_feature_columns = [
        column for column in featureColumns if column not in df_training_dataset.columns
    ]
    if missing_feature_columns:
        raise ValueError(f"Missing feature columns: {missing_feature_columns}")

    if trainRatio <= 0 or trainRatio >= 1:
        raise ValueError("trainRatio must be greater than 0 and less than 1.")
    if splitGapDays < 0:
        raise ValueError("splitGapDays must be greater than or equal to 0.")

    df_sorted = df_training_dataset.copy()
    df_sorted[dateColumn] = pandas.to_datetime(df_sorted[dateColumn])
    df_sorted = df_sorted.sort_values([dateColumn, "Maquina"]).reset_index(drop=True)

    unique_dates = pandas.Series(df_sorted[dateColumn].drop_duplicates()).sort_values()
    split_index = int(len(unique_dates) * trainRatio)
    split_index = max(1, min(split_index, len(unique_dates) - 1))
    split_date = pandas.Timestamp(unique_dates.iloc[split_index])
    test_start_date = split_date + pandas.Timedelta(days=splitGapDays)

    df_train = df_sorted[df_sorted[dateColumn] < split_date].copy()
    df_gap = df_sorted[
        (df_sorted[dateColumn] >= split_date)
        & (df_sorted[dateColumn] < test_start_date)
    ].copy()
    df_test = df_sorted[df_sorted[dateColumn] >= test_start_date].copy()

    if df_test.empty:
        raise ValueError(
            f"Temporal split produced an empty test set. Reduce splitGapDays={splitGapDays} or trainRatio={trainRatio}."
        )

    X_train = df_train[featureColumns].copy()
    y_train = df_train[[targetColumn]].copy()
    X_test = df_test[featureColumns].copy()
    y_test = df_test[[targetColumn]].copy()

    summary = pandas.DataFrame(
        [
            {
                "TargetColumn": targetColumn,
                "DateColumn": dateColumn,
                "TrainRatio": trainRatio,
                "SplitDate": split_date,
                "SplitGapDays": splitGapDays,
                "GapRows": df_gap.shape[0],
                "GapStartDate": df_gap[dateColumn].min()
                if not df_gap.empty
                else "",
                "GapEndDate": df_gap[dateColumn].max()
                if not df_gap.empty
                else "",
                "TrainRows": X_train.shape[0],
                "TestRows": X_test.shape[0],
                "FeatureColumns": len(featureColumns),
                "TrainStartDate": df_train[dateColumn].min(),
                "TrainEndDate": df_train[dateColumn].max(),
                "TestStartDate": df_test[dateColumn].min(),
                "TestEndDate": df_test[dateColumn].max(),
            }
        ]
    )

    PrintSuccess("Temporal train/test split was created.", "Temporal Split")
    PrintData(summary)
    PrintShowSeparator()
    dataset_split = DatasetSplit(
        Train=Dataset(
            X=X_train.reset_index(drop=True),
            y=y_train.reset_index(drop=True),
        ),
        Test=Dataset(
            X=X_test.reset_index(drop=True),
            y=y_test.reset_index(drop=True),
        ),
    )
    ShowDatasetSplitHead(dataset_split, f"{targetColumn} Temporal Split", headQty=5)

    return dataset_split


def DetectFeatureColumnTypes(XTrain: pandas.DataFrame) -> FeatureColumnTypes:
    """Detect categorical and numeric feature columns from the training dataset."""
    categoricalFeatures = XTrain.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numericFeatures = [
        column for column in XTrain.columns if column not in categoricalFeatures
    ]

    summary = pandas.DataFrame(
        [
            {
                "FeatureType": "Categorical",
                "ColumnCount": len(categoricalFeatures),
                "Columns": ", ".join(categoricalFeatures),
            },
            {
                "FeatureType": "Numeric",
                "ColumnCount": len(numericFeatures),
                "Columns": ", ".join(numericFeatures),
            },
        ]
    )
    PrintSuccess("Feature column types were detected.", "Feature Types")
    PrintData(summary)
    PrintShowSeparator()

    return {
        "CategoricalFeatures": categoricalFeatures,
        "NumericFeatures": numericFeatures,
    }


def BuildModelPipelines(
    categoricalFeatures: list[str],
    numericFeatures: list[str],
    enabledModelNames: list[str] | None = None,
) -> list[ModelPipelineInfo]:
    """Build all configured model pipelines."""
    if enabledModelNames is None:
        enabledModelNames = list(AVAILABLE_MODEL_FACTORIES.keys())
    enabledModelNames = RemoveDuplicatedModelNames(enabledModelNames)

    unknownModelNames = [
        modelName
        for modelName in enabledModelNames
        if modelName not in AVAILABLE_MODEL_FACTORIES
    ]
    if unknownModelNames:
        availableModelNames = ", ".join(AVAILABLE_MODEL_FACTORIES.keys())
        raise ValueError(
            f"Unknown model names: {unknownModelNames}. Available models: {availableModelNames}"
        )

    modelDefinitions: list[ClassificationModelDefinition] = [
        AVAILABLE_MODEL_FACTORIES[modelName]()
        for modelName in enabledModelNames
    ]

    modelPipelines: list[ModelPipelineInfo] = []
    for modelDefinition in modelDefinitions:
        modelPipelines.append(
            {
                "Name": modelDefinition.Name,
                "Pipeline": modelDefinition.CreatePipeline(
                    categoricalFeatures,
                    numericFeatures,
                ),
                "ParamGrid": modelDefinition.CreateParamGrid(),
            }
        )

    summary = pandas.DataFrame(
        [
            {
                "ModelName": modelPipeline["Name"],
                "CategoricalFeatures": len(categoricalFeatures),
                "NumericFeatures": len(numericFeatures),
                "ParamGridCandidates": CountGridSearchCandidates(
                    modelPipeline["ParamGrid"]
                ),
                "PipelineSteps": "preprocessor -> model",
            }
            for modelPipeline in modelPipelines
        ]
    )
    PrintSuccess(
        f"Model pipelines were created for: {', '.join(enabledModelNames)}.",
        "Model Pipelines",
    )
    PrintData(summary)
    PrintShowSeparator()

    return modelPipelines


def CountGridSearchCandidates(paramGrid: GridSearchParamGrid) -> int:
    """Count how many hyperparameter combinations will be tested."""
    return len(list(ParameterGrid(paramGrid)))


def IsBetterTargetWinner(
    candidateModel: TrainedModelInfo,
    currentWinner: TrainedModelInfo | None,
) -> bool:
    """Choose the target winner using TestF1 and conservative ensemble tie handling."""
    if currentWinner is None:
        return True

    candidateTestF1 = float(candidateModel["TestF1"])
    currentTestF1 = float(currentWinner["TestF1"])
    candidateIsEnsemble = candidateModel["ModelName"] == WeightedSoftVotingClassifier.Name
    currentIsEnsemble = currentWinner["ModelName"] == WeightedSoftVotingClassifier.Name

    if candidateTestF1 > currentTestF1:
        return True

    if candidateTestF1 < currentTestF1:
        return False

    if candidateIsEnsemble and not currentIsEnsemble:
        return False

    if not candidateIsEnsemble and currentIsEnsemble:
        return True

    return float(candidateModel["BestCVScore"]) > float(currentWinner["BestCVScore"])


def BuildWinningResults(trainingRunResult: TrainingRunResult) -> list[dict[str, object]]:
    """Build winner labels using TestF1 and requiring ensembles to improve on ties."""
    bestModels = trainingRunResult["BestModels"]
    targetWinnerByTarget = {}

    for trainedModel in bestModels:
        targetName = trainedModel["TargetName"]
        currentTargetWinner = targetWinnerByTarget.get(targetName)
        if IsBetterTargetWinner(trainedModel, currentTargetWinner):
            targetWinnerByTarget[targetName] = trainedModel

    bestModelKeys = {
        (
            trainedModel["TargetName"],
            trainedModel["ModelName"],
            trainedModel["BestCVSplits"],
        )
        for trainedModel in bestModels
    }

    for resultRow in trainingRunResult["AllResults"]:
        resultKey = (
            resultRow["TargetName"],
            resultRow["ModelName"],
            resultRow["CVSplits"],
        )
        targetWinner = targetWinnerByTarget[resultRow["TargetName"]]
        resultRow["BestFold"] = "✅ Yes" if resultKey in bestModelKeys else ""
        resultRow["TargetWinner"] = (
            "🏆 Yes"
            if (
                resultRow["TargetName"] == targetWinner["TargetName"]
                and resultRow["ModelName"] == targetWinner["ModelName"]
                and resultRow["CVSplits"] == targetWinner["BestCVSplits"]
            )
            else ""
        )

    winnerRows = []
    for trainedModel in bestModels:
        targetWinner = targetWinnerByTarget[trainedModel["TargetName"]]
        isEnsemble = trainedModel["ModelName"] == WeightedSoftVotingClassifier.Name
        winnerRows.append(
            {
                "BestFold": "" if isEnsemble else "✅ Yes",
                "Ensemble": "🤝 Yes" if isEnsemble else "",
                "TargetWinner": "🏆 Yes" if trainedModel is targetWinner else "",
                "TargetName": trainedModel["TargetName"],
                "ModelName": trainedModel["ModelName"],
                "CVSplits": (
                    "Ensemble"
                    if isEnsemble
                    else trainedModel["BestCVSplits"]
                ),
                "PredictionThreshold": trainedModel["PredictionThreshold"],
                "BestCVScore": round(float(trainedModel["BestCVScore"]), 6),
                "TestAccuracy": round(float(trainedModel["TestAccuracy"]), 6),
                "TestPrecision": round(float(trainedModel["TestPrecision"]), 6),
                "TestRecall": round(float(trainedModel["TestRecall"]), 6),
                "TestF1": round(float(trainedModel["TestF1"]), 6),
                "TestRocAuc": round(float(trainedModel["TestRocAuc"]), 6),
                "BestParams": str(trainedModel["BestParams"]),
            }
        )

    return winnerRows


def TrainModelsWithGridSearchCV(
    temporalSplits: dict[str, DatasetSplit],
    modelPipelinesByTarget: dict[str, list[ModelPipelineInfo]],
    scoring: str = "f1",
    cvSplitOptions: list[int] | None = None,
    predictionThreshold: float = 0.5,
) -> TrainingRunResult:
    """Train all model pipelines using TimeSeriesSplit and GridSearchCV."""
    if cvSplitOptions is None:
        cvSplitOptions = [3, 4, 5, 7]
    if predictionThreshold <= 0 or predictionThreshold >= 1:
        raise ValueError("predictionThreshold must be greater than 0 and less than 1.")

    bestTrainedModels: dict[str, TrainedModelInfo] = {}
    resultRows = []
    trainingStartDateTime = datetime.now()
    trainingStartCounter = perf_counter()

    PrintSuccess(
        f"GridSearchCV training started at {trainingStartDateTime.strftime('%Y-%m-%d %H:%M:%S')}.",
        "Training Started",
    )

    for targetName, datasetSplit in temporalSplits.items():
        modelPipelines = modelPipelinesByTarget[targetName]

        for modelPipeline in modelPipelines:
            modelName = modelPipeline["Name"]
            bestModelKey = f"{targetName}_{modelName}"

            for cvSplits in cvSplitOptions:
                timeSeriesSplit = TimeSeriesSplit(n_splits=cvSplits)
                runStartDateTime = datetime.now()
                runStartCounter = perf_counter()
                PrintSuccess(
                    f"Training {modelName} for {targetName} using GridSearchCV with {cvSplits} folds. Start: {runStartDateTime.strftime('%Y-%m-%d %H:%M:%S')}.",
                    "Training Model",
                )

                gridSearch = GridSearchCV(
                    estimator=modelPipeline["Pipeline"],
                    param_grid=modelPipeline["ParamGrid"],
                    scoring=scoring,
                    cv=timeSeriesSplit,
                    n_jobs=-1,
                    refit=True,
                )
                yTrain = datasetSplit.Train.y.iloc[:, 0]
                yTest = datasetSplit.Test.y.iloc[:, 0]

                gridSearch.fit(datasetSplit.Train.X, yTrain)
                bestEstimator = gridSearch.best_estimator_
                yPredProba = bestEstimator.predict_proba(datasetSplit.Test.X)[:, 1]
                yPred = yPredProba >= predictionThreshold
                runEndDateTime = datetime.now()
                runElapsedSeconds = perf_counter() - runStartCounter

                try:
                    testRocAuc = roc_auc_score(yTest, yPredProba)
                except ValueError:
                    testRocAuc = 0.0

                testAccuracy = accuracy_score(yTest, yPred)
                testPrecision = precision_score(yTest, yPred, zero_division=0)
                testRecall = recall_score(yTest, yPred, zero_division=0)
                testF1 = f1_score(yTest, yPred, zero_division=0)

                trainedModelInfo: TrainedModelInfo = {
                    "TargetName": targetName,
                    "ModelName": modelName,
                    "BestEstimator": bestEstimator,
                    "BestParams": gridSearch.best_params_,
                    "BestCVScore": float(gridSearch.best_score_),
                    "BestCVSplits": cvSplits,
                    "PredictionThreshold": predictionThreshold,
                    "TestAccuracy": float(testAccuracy),
                    "TestPrecision": float(testPrecision),
                    "TestRecall": float(testRecall),
                    "TestF1": float(testF1),
                    "TestRocAuc": float(testRocAuc),
                }
                currentBestModel = bestTrainedModels.get(bestModelKey)
                if (
                    currentBestModel is None
                    or trainedModelInfo["BestCVScore"] > currentBestModel["BestCVScore"]
                ):
                    bestTrainedModels[bestModelKey] = trainedModelInfo

                resultRows.append(
                    {
                        "TargetName": targetName,
                        "ModelName": modelName,
                        "Scoring": scoring,
                        "PredictionThreshold": predictionThreshold,
                        "CVSplits": cvSplits,
                        "StartTime": runStartDateTime.strftime("%Y-%m-%d %H:%M:%S"),
                        "EndTime": runEndDateTime.strftime("%Y-%m-%d %H:%M:%S"),
                        "ElapsedSeconds": round(float(runElapsedSeconds), 6),
                        "BestCVScore": round(float(gridSearch.best_score_), 6),
                        "TestAccuracy": round(float(testAccuracy), 6),
                        "TestPrecision": round(float(testPrecision), 6),
                        "TestRecall": round(float(testRecall), 6),
                        "TestF1": round(float(testF1), 6),
                        "TestRocAuc": round(float(testRocAuc), 6),
                        "BestParams": str(gridSearch.best_params_),
                    }
                )

    trainingEndDateTime = datetime.now()
    trainingElapsedSeconds = perf_counter() - trainingStartCounter
    bestModels = list(bestTrainedModels.values())
    trainingRunResult: TrainingRunResult = {
        "StartedAt": trainingStartDateTime.strftime("%Y-%m-%d %H:%M:%S"),
        "EndedAt": trainingEndDateTime.strftime("%Y-%m-%d %H:%M:%S"),
        "ElapsedSeconds": round(float(trainingElapsedSeconds), 6),
        "Scoring": scoring,
        "CvSplitOptions": cvSplitOptions,
        "PredictionThreshold": predictionThreshold,
        "BestModels": bestModels,
        "AllResults": resultRows,
        "WinningResults": [],
    }
    winnerRows = BuildWinningResults(trainingRunResult)
    trainingRunResult["WinningResults"] = winnerRows

    PrintSuccess("GridSearchCV training completed.", "Training Complete")
    PrintData(
        pandas.DataFrame(
            [
                {
                    "StartTime": trainingStartDateTime.strftime("%Y-%m-%d %H:%M:%S"),
                    "EndTime": trainingEndDateTime.strftime("%Y-%m-%d %H:%M:%S"),
                    "ElapsedSeconds": round(float(trainingElapsedSeconds), 6),
                }
            ]
        )
    )
    PrintData(pandas.DataFrame(resultRows))
    PrintShowSeparator()
    ShowTitleBox(
        "🏆 Winning Models",
        max_len=120,
        color=ConsoleColor.YELLOW,
    )
    PrintData(pandas.DataFrame(winnerRows))
    PrintShowSeparator()

    return trainingRunResult


def AddWeightedSoftVotingEnsembles(
    trainingRunResult: TrainingRunResult,
    temporalSplits: dict[str, DatasetSplit],
) -> TrainingRunResult:
    """Build and evaluate a weighted soft voting ensemble for each target."""
    ensembleRows = []
    predictionThreshold = trainingRunResult["PredictionThreshold"]

    targetNames = sorted(
        {
            trainedModel["TargetName"]
            for trainedModel in trainingRunResult["BestModels"]
        }
    )

    for targetName in targetNames:
        targetModels = [
            trainedModel
            for trainedModel in trainingRunResult["BestModels"]
            if trainedModel["TargetName"] == targetName
            and trainedModel["ModelName"] != WeightedSoftVotingClassifier.Name
        ]

        if len(targetModels) < 2:
            PrintWarning(
                f"Target {targetName} needs at least two trained models to build an ensemble.",
                "Ensemble Skipped",
            )
            continue

        estimators: list[tuple[str, ProbabilisticEstimator]] = []
        for trainedModel in targetModels:
            estimator = trainedModel["BestEstimator"]
            if not hasattr(estimator, "predict_proba"):
                raise ValueError(
                    f'Model {trainedModel["ModelName"]} does not support predict_proba.'
                )

            estimators.append(
                (
                    trainedModel["ModelName"],
                    cast(ProbabilisticEstimator, estimator),
                )
            )
        weights = [
            max(float(trainedModel["BestCVScore"]), 0.000001)
            for trainedModel in targetModels
        ]
        ensembleEstimator = WeightedSoftVotingClassifier(
            estimators=estimators,
            weights=weights,
            predictionThreshold=predictionThreshold,
        )
        datasetSplit = temporalSplits[targetName]
        yTest = datasetSplit.Test.y.iloc[:, 0]
        yPredProba = ensembleEstimator.predict_proba(datasetSplit.Test.X)[:, 1]
        yPred = yPredProba >= predictionThreshold

        try:
            testRocAuc = roc_auc_score(yTest, yPredProba)
        except ValueError:
            testRocAuc = 0.0

        testAccuracy = accuracy_score(yTest, yPred)
        testPrecision = precision_score(yTest, yPred, zero_division=0)
        testRecall = recall_score(yTest, yPred, zero_division=0)
        testF1 = f1_score(yTest, yPred, zero_division=0)
        ensembleBestCVScore = sum(weights) / len(weights)

        ensembleModelInfo: TrainedModelInfo = {
            "TargetName": targetName,
            "ModelName": WeightedSoftVotingClassifier.Name,
            "BestEstimator": ensembleEstimator,
            "BestParams": ensembleEstimator.get_metadata(),
            "BestCVScore": float(ensembleBestCVScore),
            "BestCVSplits": 0,
            "PredictionThreshold": predictionThreshold,
            "TestAccuracy": float(testAccuracy),
            "TestPrecision": float(testPrecision),
            "TestRecall": float(testRecall),
            "TestF1": float(testF1),
            "TestRocAuc": float(testRocAuc),
        }
        trainingRunResult["BestModels"].append(ensembleModelInfo)

        ensembleRow = {
            "TargetName": targetName,
            "ModelName": WeightedSoftVotingClassifier.Name,
            "Scoring": trainingRunResult["Scoring"],
            "PredictionThreshold": predictionThreshold,
            "CVSplits": 0,
            "StartTime": "",
            "EndTime": "",
            "ElapsedSeconds": 0.0,
            "BestCVScore": round(float(ensembleBestCVScore), 6),
            "TestAccuracy": round(float(testAccuracy), 6),
            "TestPrecision": round(float(testPrecision), 6),
            "TestRecall": round(float(testRecall), 6),
            "TestF1": round(float(testF1), 6),
            "TestRocAuc": round(float(testRocAuc), 6),
            "BestParams": str(ensembleEstimator.get_metadata()),
            "BestFold": "",
            "Ensemble": "🤝 Yes",
            "TargetWinner": "",
        }
        trainingRunResult["AllResults"].append(ensembleRow)
        ensembleRows.append(ensembleRow)

    trainingRunResult["WinningResults"] = BuildWinningResults(trainingRunResult)

    if ensembleRows:
        ShowTitleBox(
            "🤝 Weighted Soft Voting Ensembles",
            max_len=120,
            color=ConsoleColor.YELLOW,
        )
        PrintData(pandas.DataFrame(ensembleRows))
        PrintShowSeparator()

        ShowTitleBox(
            "🏆 Winning Models With Ensembles",
            max_len=120,
            color=ConsoleColor.YELLOW,
        )
        PrintData(pandas.DataFrame(trainingRunResult["WinningResults"]))
        PrintShowSeparator()

    return trainingRunResult


def SaveTrainedModels(
    trainedModels: list[TrainedModelInfo],
    modelsDirectory: Path | None = None,
) -> pandas.DataFrame:
    """Save the best trained pipeline for each model and target."""
    if modelsDirectory is None:
        modelsDirectory = Path("Models")

    modelsDirectory.mkdir(parents=True, exist_ok=True)
    savedModelRows = []

    for trainedModel in trainedModels:
        modelName = trainedModel["ModelName"]
        targetName = trainedModel["TargetName"]
        modelFileName = f"{modelName}-{targetName}-Model.joblib"
        modelPath = modelsDirectory / modelFileName

        joblib.dump(trainedModel["BestEstimator"], modelPath)
        savedModelRows.append(
            {
                "ModelName": modelName,
                "TargetName": targetName,
                "BestCVSplits": trainedModel["BestCVSplits"],
                "PredictionThreshold": trainedModel["PredictionThreshold"],
                "BestCVScore": round(float(trainedModel["BestCVScore"]), 6),
                "TestAccuracy": round(float(trainedModel["TestAccuracy"]), 6),
                "TestPrecision": round(float(trainedModel["TestPrecision"]), 6),
                "TestRecall": round(float(trainedModel["TestRecall"]), 6),
                "TestF1": round(float(trainedModel["TestF1"]), 6),
                "TestRocAuc": round(float(trainedModel["TestRocAuc"]), 6),
                "ModelPath": str(modelPath),
                "BestParams": str(trainedModel["BestParams"]),
            }
        )

    savedModelsDataFrame = pandas.DataFrame(savedModelRows)
    PrintSuccess("Best trained models were saved.", "Models Saved")
    PrintData(savedModelsDataFrame)
    PrintShowSeparator()

    return savedModelsDataFrame


def SaveTrainingResultsJson(
    trainingRunResult: TrainingRunResult,
    savedModelsDataFrame: pandas.DataFrame,
    enabledModelNames: list[str],
    cvSplitOptions: list[int],
    targetNames: list[str],
    outputPath: Path | None = None,
) -> Path:
    """Save the latest training results, saved models, and metadata as a JSON file."""
    if outputPath is None:
        outputPath = Path("Models") / "TrainingResults.json"

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    savedModelsByKey = {
        f"{row['ModelName']}_{row['TargetName']}": row["ModelPath"]
        for row in savedModelsDataFrame.to_dict("records")
    }
    trainedModels = trainingRunResult["BestModels"]
    trainingResults = {
        "GeneratedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Description": "Latest training run results, saved model paths, metrics, winners, and configuration.",
        "Configuration": {
            "EnabledModelNames": enabledModelNames,
            "TargetNames": targetNames,
            "CvSplitOptions": cvSplitOptions,
            "Scoring": trainingRunResult["Scoring"],
            "PredictionThreshold": trainingRunResult["PredictionThreshold"],
        },
        "RunMetadata": {
            "StartedAt": trainingRunResult["StartedAt"],
            "EndedAt": trainingRunResult["EndedAt"],
            "ElapsedSeconds": trainingRunResult["ElapsedSeconds"],
        },
        "SavedModels": savedModelsDataFrame.to_dict("records"),
        "BestModels": [],
        "WinningResults": trainingRunResult["WinningResults"],
        "AllResults": trainingRunResult["AllResults"],
    }

    for trainedModel in trainedModels:
        modelName = trainedModel["ModelName"]
        targetName = trainedModel["TargetName"]
        trainingResults["BestModels"].append(
            {
                "ModelName": modelName,
                "TargetName": targetName,
                "BestCVSplits": trainedModel["BestCVSplits"],
                "PredictionThreshold": trainedModel["PredictionThreshold"],
                "BestCVScore": round(float(trainedModel["BestCVScore"]), 6),
                "TestAccuracy": round(float(trainedModel["TestAccuracy"]), 6),
                "TestPrecision": round(float(trainedModel["TestPrecision"]), 6),
                "TestRecall": round(float(trainedModel["TestRecall"]), 6),
                "TestF1": round(float(trainedModel["TestF1"]), 6),
                "TestRocAuc": round(float(trainedModel["TestRocAuc"]), 6),
                "BestParams": trainedModel["BestParams"],
                "ModelPath": savedModelsByKey.get(f"{modelName}_{targetName}", ""),
            }
        )

    outputPath.write_text(
        json.dumps(trainingResults, indent=4, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    PrintSuccess(f'Training results were saved to "{outputPath}".', "Results Saved")
    PrintShowSeparator()

    return outputPath


def main(
    enabledModelNames: list[str] | None = None,
    cvSplitOptions: list[int] | None = None,
    targetNames: list[str] | None = None,
    predictionThreshold: float | None = None,
):
    """Run the model training workflow."""
    if enabledModelNames is None:
        enabledModelNames = GetDefaultEnabledModelNames()
    if cvSplitOptions is None:
        cvSplitOptions = GetDefaultCvSplitOptions()
    if targetNames is None:
        targetNames = GetDefaultTargetNames()
    if predictionThreshold is None:
        predictionThreshold = GetDefaultPredictionThreshold()

    ShowTrainBanner()
    ShowTitleBoxWithSpacing(
        "ℹ️ Predictive Maintenance Ensemble Model - Training",
        max_len=120,
    )
    df_clean = LoadAndInspectDataset()

    ShowTitleBox(
        "📅 Daily Failures Dataset",
        max_len=120,
        color=ConsoleColor.BLUE,
    )
    df_daily_failures = BuildDailyFailuresDataFrame(df_clean)

    ShowTitleBox(
        "🗓️ Machine Daily Series Dataset",
        max_len=120,
        color=ConsoleColor.BLUE,
    )
    df_machine_daily_series = BuildMachineDailySeriesDataFrame(
        df_clean,
        df_daily_failures,
    )

    ShowTitleBox(
        "📈 Historical Failure Features",
        max_len=120,
        color=ConsoleColor.BLUE,
    )
    df_machine_daily_series_with_history = AddHistoricalFailureFeatures(
        df_machine_daily_series,
        df_clean,
    )

    ShowTitleBox(
        "🎯 Failure Target Columns",
        max_len=120,
        color=ConsoleColor.BLUE,
    )
    df_training_dataset = AddFailureTargetColumns(df_machine_daily_series_with_history)

    ShowTitleBox(
        "🧩 Feature And Target Column Sets",
        max_len=120,
        color=ConsoleColor.BLUE,
    )
    training_column_sets = BuildFeatureTargetColumnSets(df_training_dataset)
    targetNames = ValidateTargetNames(targetNames, training_column_sets)

    ShowTitleBox(
        "✂️ Temporal Train Test Split",
        max_len=120,
        color=ConsoleColor.BLUE,
    )
    temporal_splits: dict[str, DatasetSplit] = {}
    for splitName in targetNames:
        columnSet = training_column_sets[splitName]
        splitGapDays = GetSplitGapDaysByTargetName(splitName)
        temporal_splits[splitName] = BuildTemporalTrainTestSplit(
            df_training_dataset,
            featureColumns=columnSet["FeatureColumns"],
            targetColumn=columnSet["TargetColumn"],
            dateColumn="Fecha",
            trainRatio=0.8,
            splitGapDays=splitGapDays,
        )

    ShowTitleBox(
        "🧠 Model Pipelines",
        max_len=120,
        color=ConsoleColor.BLUE,
    )
    modelPipelinesByTarget: dict[str, list[ModelPipelineInfo]] = {}
    for splitName in targetNames:
        ShowTitleBox(
            f"🧠 Model Pipelines - {splitName}",
            max_len=120,
            color=ConsoleColor.MAGENTA,
        )
        featureTypes = DetectFeatureColumnTypes(temporal_splits[splitName].Train.X)
        modelPipelinesByTarget[splitName] = BuildModelPipelines(
            categoricalFeatures=featureTypes["CategoricalFeatures"],
            numericFeatures=featureTypes["NumericFeatures"],
            enabledModelNames=enabledModelNames,
        )

    ShowTitleBox(
        "🏋️ GridSearchCV Model Training",
        max_len=120,
        color=ConsoleColor.BLUE,
    )
    trainingRunResult = TrainModelsWithGridSearchCV(
        temporalSplits=temporal_splits,
        modelPipelinesByTarget=modelPipelinesByTarget,
        scoring="f1",
        cvSplitOptions=cvSplitOptions,
        predictionThreshold=predictionThreshold,
    )
    trainingRunResult = AddWeightedSoftVotingEnsembles(
        trainingRunResult,
        temporal_splits,
    )
    trainedModels = trainingRunResult["BestModels"]

    ShowTitleBox(
        "💾 Save Best Trained Models",
        max_len=120,
        color=ConsoleColor.BLUE,
    )
    savedModelsDataFrame = SaveTrainedModels(trainedModels)
    SaveTrainingResultsJson(
        trainingRunResult,
        savedModelsDataFrame,
        enabledModelNames=enabledModelNames,
        cvSplitOptions=cvSplitOptions,
        targetNames=targetNames,
    )


if __name__ == "__main__":
    main()
