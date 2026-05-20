"""Utility helpers for console output, environment diagnostics, and dependencies."""

import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


CORE_DIR = Path(__file__).resolve().parent
SRC_DIR = CORE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
REQUIREMENTS_PATH = SRC_DIR / "Requirements.txt"
BASE_DEPENDENCIES = [
    line.strip()
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]


class ConsoleColor(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


@dataclass(frozen=True)
class BoxStyle:
    top_left: str
    top_right: str
    bottom_left: str
    bottom_right: str
    horizontal: str
    vertical: str


class TitleBoxLineStyle(Enum):
    SIMPLE = BoxStyle("┌", "┐", "└", "┘", "─", "│")
    DOUBLE = BoxStyle("╔", "╗", "╚", "╝", "═", "║")
    ROUNDED = BoxStyle("╭", "╮", "╰", "╯", "─", "│")
    HEAVY = BoxStyle("┏", "┓", "┗", "┛", "━", "┃")
    ASCII = BoxStyle("+", "+", "+", "+", "-", "|")
    DOUBLE_BOLD = BoxStyle("╔", "╗", "╚", "╝", "╬", "║")
    BLOCK = BoxStyle("█", "█", "█", "█", "█", "█")
    HEAVY_CROSS = BoxStyle("╒", "╕", "╘", "╛", "╪", "┃")
    METAL = BoxStyle("╞", "╡", "╘", "╛", "═", "║")


def PrintColor(message: str, color: ConsoleColor) -> str:
    return f"{color.value}{message}{ConsoleColor.RESET.value}"


def PrintMessage(
    message: str,
    title: str = "Message",
    color: ConsoleColor = ConsoleColor.WHITE,
    icon: str = "💬",
    end: str = "\n",
) -> None:
    print(f"{PrintColor(icon + ' ' + title.upper() + ':', color)} {message}", end=end)


def PrintInfo(message: str, title: str = "Info", end: str = "\n") -> None:
    PrintMessage(message, title, ConsoleColor.CYAN, "ℹ️", end)


def PrintError(message: str, title: str = "Error", end: str = "\n") -> None:
    PrintMessage(message, title, ConsoleColor.RED, "❌", end)


def PrintWarning(message: str, title: str = "Warning", end: str = "\n") -> None:
    PrintMessage(message, title, ConsoleColor.YELLOW, "⚠️", end)


def PrintSuccess(message: str, title: str = "Success", end: str = "\n") -> None:
    PrintMessage(message, title, ConsoleColor.GREEN, "✅", end)


def GetVisibleLength(text: str) -> int:
    try:
        from wcwidth import wcswidth

        visible_length = wcswidth(text)
        return visible_length if visible_length >= 0 else len(text)
    except Exception:
        return len(text)


def ShowTitleBox(
    text: str,
    max_len: int = 100,
    box_line_style: TitleBoxLineStyle = TitleBoxLineStyle.BLOCK,
    color: ConsoleColor = ConsoleColor.CYAN,
) -> None:
    style = box_line_style.value
    text_len = GetVisibleLength(text)
    inner = max(max_len, text_len)
    left = (inner - text_len) // 2
    right = inner - text_len - left

    top = f"{style.top_left}{style.horizontal * (inner + 2)}{style.top_right}"
    middle = f"{style.vertical} {' ' * left}{text}{' ' * right} {style.vertical}"
    bottom = f"{style.bottom_left}{style.horizontal * (inner + 2)}{style.bottom_right}"

    print(PrintColor("\n".join([top, middle, bottom]), color))


def ShowTitleBoxWithSpacing(
    text: str,
    max_len: int = 100,
    box_line_style: TitleBoxLineStyle = TitleBoxLineStyle.BLOCK,
    color: ConsoleColor = ConsoleColor.CYAN,
) -> None:
    style = box_line_style.value
    text_len = GetVisibleLength(text)
    inner = max(max_len, text_len)
    left = (inner - text_len) // 2
    right = inner - text_len - left

    top = f"{style.top_left}{style.horizontal * (inner + 2)}{style.top_right}"
    empty = f"{style.vertical}{' ' * (inner + 2)}{style.vertical}"
    middle = f"{style.vertical} {' ' * left}{text}{' ' * right} {style.vertical}"
    bottom = f"{style.bottom_left}{style.horizontal * (inner + 2)}{style.bottom_right}"

    print(PrintColor("\n".join([top, empty, middle, empty, bottom]), color))


def RunCommand(
    command_list: list[str],
    print_command: bool = True,
    print_error: bool = True,
) -> subprocess.CompletedProcess[str]:
    if print_command:
        PrintInfo(" ".join(command_list), "Command")

    process = subprocess.Popen(
        command_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    stdout_lines: list[str] = []

    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
            stdout_lines.append(line)

    process.wait()

    stderr_text = ""
    if process.stderr is not None:
        stderr_text = process.stderr.read() or ""

    if process.returncode != 0 and print_error and stderr_text:
        PrintError(stderr_text, "", end="")

    return subprocess.CompletedProcess(
        args=command_list,
        returncode=process.returncode,
        stdout="".join(stdout_lines),
        stderr=stderr_text,
    )


def InstallDeps(libs: Optional[list[str]] = None) -> None:
    dependencies = libs or []

    PrintInfo("Installing base dependencies.")
    RunCommand(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        print_command=False,
        print_error=True,
    )

    if len(dependencies) == 0:
        PrintWarning("No dependencies were provided.")
        return

    result = RunCommand(
        [sys.executable, "-m", "pip", "install", *dependencies],
        print_command=False,
        print_error=True,
    )

    if result.returncode == 0:
        PrintSuccess("Dependencies installed successfully.")
    else:
        PrintError("Dependency installation finished with errors.")


def ShowEnvironmentInfo() -> None:
    PrintInfo("Environment Info")
    print("Python Version:", sys.version)
    print("Platform:", sys.platform)
    print("Executable Path:", sys.executable)
    print("Current Working Directory:", os.getcwd())
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
    print("sys.prefix:", sys.prefix)
    print("sys.base_prefix:", sys.base_prefix)
    print()
