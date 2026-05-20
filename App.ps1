if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & python -m venv .venv
}

& .venv\Scripts\python.exe Src\App.py @args
