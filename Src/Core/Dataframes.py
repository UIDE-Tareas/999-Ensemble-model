"""DataFrame helpers for loading, inspecting, cleaning, splitting, and scaling datasets."""

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import Console
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

pd.set_option("display.float_format", "{:.2f}".format)
pandas.set_option("display.max_rows", None)
pandas.set_option("display.max_columns", None)

RANDOM_STATE = 216
TRAIN_RATIO = 0.80
TEST_RATIO = 0.20
CORRELATION_THRESHOLD = 0.30
SHOW_PLOTS = True
SAVE_PLOTS = True
CONSOLE = Console()


def PrintData(value: Any = ""):
    if isinstance(value, pandas.DataFrame):
        PrintDataFrameTable(value)
    elif isinstance(value, pandas.Series):
        PrintDataFrameTable(value.to_frame())
    else:
        print(value)


def PrintDataFrameTable(df: pandas.DataFrame, showIndex: bool | None = None):
    try:
        table = Table(show_header=True, header_style="bold cyan", show_lines=True)

        if showIndex is None:
            showIndex = not isinstance(df.index, pandas.RangeIndex)

        if showIndex:
            table.add_column("Metric", overflow="fold")

        for column in df.columns:
            table.add_column(str(column), overflow="fold")

        for index, row in df.iterrows():
            values = [str(value) for value in row.tolist()]
            if showIndex:
                values = [str(index), *values]
            table.add_row(*values)

        CONSOLE.print(table)
    except Exception:
        print(df.to_string())


def PrintShowSeparator(width: int = 120, char: str = "━"):
    print()
    print(char * width)
    print()


def ShowDfInfo(df: pandas.DataFrame, title):
    print(f"ℹ️ {title} - Info")
    df.info(memory_usage="deep")
    total_bytes = df.memory_usage(deep=True).sum()
    print(
        "Deep memory total: "
        f"{total_bytes} bytes | "
        f"{total_bytes / 1024:.6f} KB | "
        f"{total_bytes / (1024**2):.6f} MB"
    )
    PrintShowSeparator()


def ShowDfHead(df: pandas.DataFrame, title: str, headQty=10):
    print(f"ℹ️ {title} - Head: First {headQty} rows.")
    PrintData(df.head(headQty))
    PrintShowSeparator()


def ShowDfTail(df: pandas.DataFrame, title: str, tailQty=10):
    print(f"ℹ️ {title} - Tail: Last {tailQty} rows.")
    PrintData(df.tail(tailQty))
    PrintShowSeparator()


def ShowDfShape(df: pandas.DataFrame, title: str):
    print(f"ℹ️ {title} - Data shape")
    print(f"{df.shape[0]} rows x {df.shape[1]} columns")
    PrintShowSeparator()


def ShowDfDataTypes(df: pandas.DataFrame, title: str):
    print(f"ℹ️ {title} - Data types")
    dtypes_df = df.dtypes.reset_index()
    dtypes_df.columns = ["Column", "Data_Type"]
    dtypes_df["Non_Null_Count"] = df.notnull().sum().values
    dtypes_df["Null_Count"] = df.isnull().sum().values
    dtypes_df["Unique_Count"] = df.nunique(dropna=True).values
    PrintData(dtypes_df)
    PrintShowSeparator()


def ShowDfMemoryUsage(df: pandas.DataFrame, title: str):
    print(f"ℹ️ {title} - Memory usage")

    def FormatFixedNumber(
        value: int | float,
        integerWidth: int = 10,
        decimalWidth: int = 10,
    ) -> str:
        totalWidth = integerWidth + decimalWidth + 1
        return f"{float(value):{totalWidth}.{decimalWidth}f}"

    memory_usage = df.memory_usage(deep=True)
    memory_df = memory_usage.reset_index()
    memory_df.columns = ["Column", "MemoryBytes"]
    memory_df["MemoryKB"] = memory_df["MemoryBytes"] / 1024
    memory_df["MemoryMB"] = memory_df["MemoryBytes"] / (1024**2)
    memory_df["MemoryBytes"] = memory_df["MemoryBytes"].apply(FormatFixedNumber)
    memory_df["MemoryKB"] = memory_df["MemoryKB"].apply(FormatFixedNumber)
    memory_df["MemoryMB"] = memory_df["MemoryMB"].apply(FormatFixedNumber)
    PrintData(memory_df)
    total_bytes = memory_usage.sum()
    print(
        "Total memory: "
        f"{FormatFixedNumber(total_bytes)} bytes | "
        f"{FormatFixedNumber(total_bytes / 1024)} KB | "
        f"{FormatFixedNumber(total_bytes / (1024**2))} MB"
    )
    PrintShowSeparator()


def ShowDfDateRanges(df: pandas.DataFrame, title: str):
    print(f"ℹ️ {title} - Date ranges")
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"])

    if datetime_cols.empty:
        print("No datetime columns found.")
        PrintShowSeparator()
        return

    rows = []
    for col in datetime_cols.columns:
        rows.append(
            {
                "Column": col,
                "Min_Date": datetime_cols[col].min(),
                "Max_Date": datetime_cols[col].max(),
                "Non_Null_Count": datetime_cols[col].notnull().sum(),
            }
        )

    PrintData(pandas.DataFrame(rows))
    PrintShowSeparator()


def ShowDfStats(df: pandas.DataFrame, title: str = ""):
    print(f"ℹ️ {title} - Descriptive statistics")
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        print("    NUMERIC COLUMNS")
        numeric_desc = numeric_cols.describe().round(2).T
        numeric_desc["var"] = numeric_cols.var(numeric_only=True).round(2)
        PrintDataFrameTable(numeric_desc.T, showIndex=True)

    non_numeric_cols = df.select_dtypes(
        include=["boolean", "string", "category", "object"]
    )
    if not non_numeric_cols.empty:
        print("    NON-NUMERIC COLUMNS")
        non_numeric_desc = non_numeric_cols.describe()
        PrintDataFrameTable(non_numeric_desc, showIndex=True)

    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"])
    if not datetime_cols.empty:
        print("    DATETIME COLUMNS")
        datetime_desc = datetime_cols.describe()
        PrintDataFrameTable(datetime_desc, showIndex=True)
    PrintShowSeparator()


def ShowDfDuplicates(
    df: pandas.DataFrame,
    title: str,
    qty: int = 10,
):
    print(f"ℹ️ {title} - DataFrame duplicates")

    dup_mask = df.duplicated()
    dup_df = df[dup_mask]
    dup_count = dup_df.shape[0]

    PrintData(
        pandas.DataFrame(
            [
                {
                    "DuplicateRowCount": dup_count,
                    "RowsShown": min(dup_count, qty),
                    "HasDuplicates": dup_count > 0,
                }
            ]
        )
    )

    if dup_count == 0:
        print("No duplicates found.")
        PrintShowSeparator()
        return

    print(f"Showing up to {qty} duplicate rows:")
    PrintData(dup_df.head(qty))

    remaining = dup_count - qty
    if remaining > 0:
        print(f"There are {remaining} additional duplicate rows not shown.")

    PrintShowSeparator()


def ShowDfUniqueValues(df: pandas.DataFrame, title: str, qty: int = 30):
    print(f"ℹ️ {title} - Unique values by column")

    table = Table(show_header=True, header_style="bold cyan", show_lines=True)
    table.add_column("Column", overflow="fold")
    table.add_column("UniqueCount", justify="right")
    table.add_column("ShownValues", justify="right")
    table.add_column("IsTruncated", justify="center")
    table.add_column("SampleValues", overflow="fold")

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        count = len(unique_vals)
        sample_values = sorted(unique_vals[:qty])

        table.add_row(
            str(col),
            str(count),
            str(min(count, qty)),
            str(count > qty),
            ", ".join([str(value) for value in sample_values]),
        )

    CONSOLE.print(table)
    PrintShowSeparator()


def ShowFullDfOverview(
    df,
    title,
    headQty=5,
    tailQty=5,
    duplicatesqty=10,
    uniqueqty=30,
):
    ShowDfInfo(df, title)
    ShowDfStats(df, title)
    ShowDfShape(df, title)
    ShowDfDuplicates(df, title, qty=duplicatesqty)
    ShowDfUniqueValues(df, title, qty=uniqueqty)
    ShowDfHead(df, title, headQty=headQty)
    ShowDfTail(df, title, tailQty=tailQty)


def ShowDfNanValues(df: pandas.DataFrame, title: str):
    print(f"ℹ️ {title} - Null value count")
    nulls_count = df.isnull().sum()
    nulls_df = nulls_count.reset_index()
    nulls_df.columns = ["Column", "Null_Count"]
    PrintData(nulls_df)
    PrintShowSeparator()


class CorrelationType(Enum):
    ALL = "all"
    STRONG = "strong"
    WEAK = "weak"


def ShowDfCorrelation(
    df: pandas.DataFrame,
    title: str,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    level: CorrelationType = CorrelationType.ALL,
    umbral: float = 0.6,
    showTable: bool = False,
    figsize: tuple = (8, 6),
    annotate: bool = True,
    outputPath: Optional[Path] = None,
    showPlot: bool = SHOW_PLOTS,
    savePlot: bool = SAVE_PLOTS,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f"{title} - {level.name} Correlation")

    print(f"ℹ️ {title} - Correlation matrix ({level.name})")

    corr = df.select_dtypes(include="number").corr()

    if level == CorrelationType.STRONG:
        corr = corr.where(np.abs(corr) >= umbral)
    elif level == CorrelationType.WEAK:
        corr = corr.where((np.abs(corr) < umbral) | (corr == 1))
    elif level != CorrelationType.ALL:
        raise ValueError(f"Invalid level: {level}")

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=annotate,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation coefficient"},
        ax=ax,
    )

    subtitle = (
        "All"
        if level == CorrelationType.ALL
        else f"Strong (|r| >= {umbral})"
        if level == CorrelationType.STRONG
        else f"Weak (|r| < {umbral})"
    )

    ax.set_title(
        f"Correlation matrix ({subtitle})",
        fontsize=12,
        pad=15,
    )

    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    if savePlot and outputPath is not None:
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outputPath, dpi=150)
        print(f'Chart saved to "{outputPath.resolve()}".')
    if showPlot:
        plt.show()
    else:
        plt.close(fig)

    if showTable:
        PrintData(corr.round(3))

    PrintShowSeparator()
    return fig, corr


def NormalizeColumnNames(df: pandas.DataFrame) -> pandas.DataFrame:
    df.columns = [
        col.strip()
        .replace("(", "")
        .replace(")", "")
        .title()
        .replace(" ", "")
        .replace("_", "")
        for col in df.columns
    ]
    return df


def DropColumns(
    df: pandas.DataFrame,
    toDrop: list[str],
    inplace: bool = False,
) -> pandas.DataFrame:
    if not toDrop:
        return df
    if inplace:
        df.drop(columns=df.columns.intersection(toDrop), inplace=True)
        return df
    return df.drop(columns=df.columns.intersection(toDrop))


@dataclass
class Dataset:
    X: pandas.DataFrame
    y: pandas.DataFrame


@dataclass
class DatasetSplit:
    Train: Dataset
    Test: Dataset


def ShowDatasetSplitHead(split: DatasetSplit, title: str, headQty: int = 5):
    ShowDfHead(split.Train.X, f"{title} - X Train", headQty)
    ShowDfHead(split.Train.y, f"{title} - y Train", headQty)
    ShowDfHead(split.Test.X, f"{title} - X Test", headQty)
    ShowDfHead(split.Test.y, f"{title} - y Test", headQty)


def ShowDatasetInfo(data: Dataset, title):
    titleAux = title
    title = f"{titleAux} - Features - X"
    ShowDfInfo(data.X, title)
    ShowDfShape(data.X, title)
    ShowDfStats(data.X, title)
    ShowDfNanValues(data.X, title)
    ShowDfHead(data.X, title)
    ShowDfTail(data.X, title)

    title = f"{titleAux} - Target - y"
    ShowDfInfo(data.y, title)
    ShowDfShape(data.y, title)
    ShowDfStats(data.y, title)
    ShowDfNanValues(data.y, title)
    ShowDfHead(data.y, title)
    ShowDfTail(data.y, title)


def ShowDatasetSplitInfo(split: DatasetSplit, title: str, headQty: int = 5):
    titleAux = title
    title = f"{titleAux} - TRAIN"
    ShowDatasetInfo(split.Train, title)
    title = f"{titleAux} - TEST"
    ShowDatasetInfo(split.Test, title)


def SplitDataset(
    data: Dataset,
    trainRatio: float = TRAIN_RATIO,
    testRatio: float = TEST_RATIO,
    randomState: int = RANDOM_STATE,
) -> DatasetSplit:
    if round(trainRatio + testRatio, 10) != 1:
        raise ValueError("trainRatio and testRatio must add up to 1.")

    XTrain, XTest, yTrain, yTest = train_test_split(
        data.X,
        data.y,
        train_size=trainRatio,
        test_size=testRatio,
        random_state=randomState,
    )
    return DatasetSplit(
        Train=Dataset(X=XTrain.reset_index(drop=True), y=yTrain.reset_index(drop=True)),
        Test=Dataset(X=XTest.reset_index(drop=True), y=yTest.reset_index(drop=True)),
    )


class ScalerProtocol(Protocol):
    def fit(self, X, y: Any = None) -> Any: ...
    def transform(self, X) -> Any: ...
    def fit_transform(self, X, y: Any = None) -> Any: ...


@dataclass
class ScaledDatasetSplit(DatasetSplit):
    Scaler: ScalerProtocol


class ScalerType(Enum):
    STANDARD = "Standard"
    MIN_MAX = "minmax"
    ROBUST = "robust"
    MAX_ABS = "maxabs"
    NORMALIZER = "normalizer"
    QUANTILE = "quantile"
    POWER = "power"
    FUNCTION = "function"


def CreateScaler(scalerType: ScalerType, **kwargs) -> ScalerProtocol:
    if scalerType == ScalerType.STANDARD:
        return StandardScaler(**kwargs)
    if scalerType == ScalerType.MIN_MAX:
        return MinMaxScaler(**kwargs)
    if scalerType == ScalerType.ROBUST:
        return RobustScaler(**kwargs)
    if scalerType == ScalerType.MAX_ABS:
        return MaxAbsScaler(**kwargs)
    if scalerType == ScalerType.NORMALIZER:
        return Normalizer(**kwargs)
    if scalerType == ScalerType.QUANTILE:
        return QuantileTransformer(**kwargs)
    if scalerType == ScalerType.POWER:
        return PowerTransformer(**kwargs)
    if scalerType == ScalerType.FUNCTION:
        return FunctionTransformer(**kwargs)
    raise ValueError(f"Unsupported ScalerType: {scalerType}")


def DetectScaler(scaler: ScalerProtocol) -> ScalerType:
    if isinstance(scaler, StandardScaler):
        return ScalerType.STANDARD
    if isinstance(scaler, MinMaxScaler):
        return ScalerType.MIN_MAX
    if isinstance(scaler, RobustScaler):
        return ScalerType.ROBUST
    if isinstance(scaler, MaxAbsScaler):
        return ScalerType.MAX_ABS
    if isinstance(scaler, Normalizer):
        return ScalerType.NORMALIZER
    if isinstance(scaler, QuantileTransformer):
        return ScalerType.QUANTILE
    if isinstance(scaler, PowerTransformer):
        return ScalerType.POWER
    if isinstance(scaler, FunctionTransformer):
        return ScalerType.FUNCTION
    raise ValueError(f"Unrecognized scaler type: {type(scaler)}")


def ScaleDatasetSplit(
    split: DatasetSplit,
    scaler: ScalerProtocol = StandardScaler(),
) -> ScaledDatasetSplit:
    XTrainScaledValues = scaler.fit_transform(split.Train.X)
    XTestScaledValues = scaler.transform(split.Test.X)

    XTrainScaled = pandas.DataFrame(
        XTrainScaledValues,
        columns=split.Train.X.columns,
        index=split.Train.X.index,
    )

    XTestScaled = pandas.DataFrame(
        XTestScaledValues,
        columns=split.Test.X.columns,
        index=split.Test.X.index,
    )

    trainScaledDataset = Dataset(X=XTrainScaled, y=split.Train.y.copy())
    testScaledDataset = Dataset(X=XTestScaled, y=split.Test.y.copy())

    return ScaledDatasetSplit(
        Train=trainScaledDataset,
        Test=testScaledDataset,
        Scaler=scaler,
    )


@dataclass
class PcaDatasetSplit(DatasetSplit):
    Pca: PCA
    Scaler: ScalerProtocol | None = None


def ApplyPCA(
    split: ScaledDatasetSplit,
    explainedVarianceRatioSum: float = 0.95,
    randomState: int = RANDOM_STATE,
) -> PcaDatasetSplit:
    def GetPCNames(n: int) -> list[str]:
        return [f"PC{i}" for i in range(1, n + 1)]

    pca = PCA(n_components=explainedVarianceRatioSum, random_state=randomState)

    XTrainPCA = pca.fit_transform(split.Train.X)
    XTestPCA = pca.transform(split.Test.X)

    XTrainPcaDf = pandas.DataFrame(
        XTrainPCA,
        index=split.Train.X.index,
        columns=GetPCNames(XTrainPCA.shape[1]),
    )

    XTestPcaDf = pandas.DataFrame(
        XTestPCA,
        index=split.Test.X.index,
        columns=GetPCNames(XTestPCA.shape[1]),
    )

    return PcaDatasetSplit(
        Train=Dataset(X=XTrainPcaDf, y=split.Train.y.copy()),
        Test=Dataset(X=XTestPcaDf, y=split.Test.y.copy()),
        Pca=pca,
        Scaler=split.Scaler,
    )


SplitLike = ScaledDatasetSplit | PcaDatasetSplit


@dataclass(frozen=True)
class SplitTypeInfo:
    IsPCA: bool
    IsScaled: bool
    IsRaw: bool


def DetectSplitType(split) -> SplitTypeInfo:
    isPca = isinstance(split, PcaDatasetSplit)
    isScaled = isinstance(split, ScaledDatasetSplit)
    isRaw = not isPca and not isScaled

    return SplitTypeInfo(
        IsPCA=isPca,
        IsScaled=isScaled,
        IsRaw=isRaw,
    )


def RemoveDfDuplicates(
    df: pandas.DataFrame,
    inplace: bool = False,
) -> pandas.DataFrame:
    if inplace:
        df.drop_duplicates(inplace=True)
        return df
    return df.drop_duplicates()


def RemoveDfNanValues(
    df: pandas.DataFrame,
    inplace: bool = False,
) -> pandas.DataFrame:
    if inplace:
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    return df.dropna().reset_index(drop=True)


def ShowDfNanRows(
    df: pandas.DataFrame,
    title: str,
    qty: int = 30,
):
    print(f"ℹ️ {title} - Rows with null values")
    nan_rows = df[df.isnull().any(axis=1)]
    nan_count = nan_rows.shape[0]

    PrintData(
        pandas.DataFrame(
            [
                {
                    "NullRowCount": nan_count,
                    "RowsShown": min(nan_count, qty),
                    "HasNullRows": nan_count > 0,
                }
            ]
        )
    )

    if nan_count > 0:
        PrintData(nan_rows.head(qty))

    PrintShowSeparator()
