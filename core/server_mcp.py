from __future__ import annotations
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
from .dataset_inspector import DatasetInspector

# -------------------------------------------------------------------
# Server setup (MCP stdio)
# -------------------------------------------------------------------
APP_NAME = "dataset-inspector"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

mcp = FastMCP(APP_NAME)

# Persistent inspector
_INSPECTOR: Optional[DatasetInspector] = None

def _require_inspector() -> DatasetInspector:
    if _INSPECTOR is None:
        raise RuntimeError("No active dataset. Call switch_dataset() first.")
    return _INSPECTOR

# -------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------
@mcp.tool()
def switch_dataset(file_path: str) -> dict:
    """
    Set or change the active dataset.
    file_path is relative to the project root (e.g. 'uploads/data.csv').
    """
    global _INSPECTOR
    p = (PROJECT_ROOT / file_path).resolve()
    if not p.exists():
        return {"error": f"File not found: {p}", "active_dataset": None, "columns": {}}
    try:
        _INSPECTOR = DatasetInspector(p)
        cols = _INSPECTOR.list_columns()
        return {"active_dataset": str(p), "columns": cols}
    except Exception as e:
        return {"error": str(e), "active_dataset": None, "columns": {}}

@mcp.tool()
def get_data_overview() -> dict:
    try:
        di = _require_inspector()
        result = di.get_data_overview()
        return result
    except RuntimeError as e:
        return {"error": str(e)}

@mcp.tool()
def list_columns() -> dict:
    try:
        di = _require_inspector()
        result = di.list_columns()
        return result  # Already returns dict with columns, total_rows, etc.
    except RuntimeError as e:
        return {"error": str(e), "columns": {}}

@mcp.tool()
def summary_stats() -> dict:
    try:
        di = _require_inspector()
        result = di.summary_stats()
        return result  # Already returns dict with stats, insights, etc.
    except RuntimeError as e:
        return {"error": str(e), "stats": {}}

@mcp.tool()
def get_column_summary(column: str) -> dict:
    try:
        di = _require_inspector()
        return {"summary": di.get_column_summary(column)}
    except (RuntimeError, ValueError) as e:
        return {"error": str(e), "summary": {}}

@mcp.tool()
def plot_column(column: str, kind: str = "auto", y: str | None = None, group_by: str | None = None, bins: int = 30) -> dict:
    try:
        di = _require_inspector()
        out = di.plot_column(column, kind=kind, y=y, group_by=group_by, bins=bins)
        return {"path": out, "column": column, "kind": kind, "y": y}
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        return {"error": error_msg, "path": None, "column": column, "kind": kind, "y": y}

@mcp.tool()
def correlation_matrix(method: str = "pearson", threshold: float = 0.5) -> dict:
    try:
        di = _require_inspector()
        result = di.correlation_matrix(method=method, threshold=threshold)
        return result
    except RuntimeError as e:
        return {"error": str(e), "matrix": {}}

@mcp.tool()
def plot_correlation_matrix() -> dict:
    try:
        di = _require_inspector()
        out = di.plot_correlation_matrix()
        # Also include correlation data so LLM can reference it
        corr_data = di.correlation_matrix()
        return {
            "path": out,
            "correlation_data_available": True,
            "columns": corr_data.get("columns", []),
            "matrix": corr_data.get("matrix", {}),
            "note": "Correlation matrix visualization created. All correlation values are displayed in the heatmap."
        }
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        return {"error": error_msg, "path": None, "traceback": traceback.format_exc()}

@mcp.tool()
def compare_columns(column1: str, column2: str) -> dict:
    try:
        di = _require_inspector()
        result = di.compare_columns(column1, column2)
        return result
    except (RuntimeError, ValueError) as e:
        return {"error": str(e)}

@mcp.tool()
def get_data_sample(n: int = 10, sample_type: str = "head") -> dict:
    try:
        di = _require_inspector()
        result = di.get_data_sample(n=n, sample_type=sample_type)
        return result
    except (RuntimeError, ValueError) as e:
        return {"error": str(e)}

@mcp.tool()
def group_by_stats(group_by: str, column: str) -> dict:
    try:
        di = _require_inspector()
        result = di.group_by_stats(group_by, column)
        return result
    except (RuntimeError, ValueError) as e:
        return {"error": str(e)}

@mcp.tool()
def detect_outliers(column: str, method: str = "iqr") -> dict:
    try:
        di = _require_inspector()
        result = di.detect_outliers(column, method)
        return result
    except (RuntimeError, ValueError) as e:
        return {"error": str(e)}

@mcp.tool()
def data_quality_report() -> dict:
    try:
        di = _require_inspector()
        result = di.data_quality_report()
        return result
    except RuntimeError as e:
        return {"error": str(e)}

# -------------------------------------------------------------------
# Run the server (stdio)
# -------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
