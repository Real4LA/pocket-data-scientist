import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_MODULE = "core.server_mcp"  # run as module for reliable imports

class ToolClient:
    """MCP stdio client that talks to our FastMCP server."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._log_path = PROJECT_ROOT / "mcp_stderr.log"
        self._initialized = False
        self._request_id = 0
        self._start_mcp()
        self._initialize_mcp()

    # ---------- MCP stdio ----------
    def _start_mcp(self):
        self._stop_mcp()
        cmd = [sys.executable, "-u", "-m", SERVER_MODULE]
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(self._log_path, "w", encoding="utf-8"),
            text=True,
            bufsize=1,
        )
        self._initialized = False
        self._request_id = 0

    def _initialize_mcp(self):
        """Perform MCP initialization handshake."""
        if self._initialized:
            return
        
        # Step 1: Send initialize request
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "dataset-inspector-client",
                "version": "1.0.0"
            }
        }
        
        try:
            result = self._send("initialize", init_params)
            # Step 2: Send initialized notification (FastMCP expects notifications/initialized)
            self._send_notification("notifications/initialized", {})
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MCP server: {e}")

    def _stop_mcp(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None

    def _send_notification(self, method: str, params: Any):
        """Send a JSON-RPC notification (no response expected)."""
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("MCP process not running")
        if self._proc.poll() is not None:
            raise RuntimeError(f"MCP process exited. See log: {self._log_path}")
        
        req = {"jsonrpc": "2.0", "method": method, "params": params}
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()

    def _send(self, method: str, params: Any) -> Any:
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            raise RuntimeError("MCP process not running")
        if self._proc.poll() is not None:
            raise RuntimeError(f"MCP process exited. See log: {self._log_path}")

        self._request_id += 1
        req_id = self._request_id
        req = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()

        # Read response (may need to skip notifications)
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"No response from MCP server. See log: {self._log_path}")
            
            resp = json.loads(line)
            # Skip notifications (no id field)
            if "id" not in resp:
                continue
            # Match our request ID
            if resp.get("id") == req_id:
                if "error" in resp and resp["error"]:
                    error_msg = resp["error"]
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", str(error_msg))
                    raise RuntimeError(f"MCP error: {error_msg}")
                return resp.get("result")

    def _rpc(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """
        Try common FastMCP wire shapes. Our server (FastMCP @tool) usually
        accepts method==tool_name with named params or with {"arguments": ...}.
        Based on MCP protocol, tools/call is the standard format.
        """
        attempts = [
            ("tools/call", {"name": tool_name, "arguments": args}),
            (tool_name, args),
            (tool_name, {"arguments": args}),
        ]
        last_err = None
        for meth, par in attempts:
            try:
                result = self._send(meth, par)
                if result is not None:
                    return result
            except (RuntimeError, json.JSONDecodeError, KeyError) as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
        raise last_err if last_err else RuntimeError(f"Failed to call tool '{tool_name}' with all attempted formats")

    # ---------- Public API ----------
    def switch_dataset(self, relative_path_from_root: str):
        return self._rpc("switch_dataset", {"file_path": relative_path_from_root})

    def get_data_overview(self):
        return self._rpc("get_data_overview", {})

    def list_columns(self):
        return self._rpc("list_columns", {})

    def summary_stats(self):
        return self._rpc("summary_stats", {})

    def get_column_summary(self, column: str):
        return self._rpc("get_column_summary", {"column": column})

    def plot_column(self, column: str, kind: str = "auto", y: Optional[str] = None, group_by: Optional[str] = None, bins: int = 30):
        payload = {"column": column, "kind": kind, "bins": bins}
        if y is not None:
            payload["y"] = y
        if group_by is not None:
            payload["group_by"] = group_by
        return self._rpc("plot_column", payload)

    def correlation_matrix(self, method: str = "pearson", threshold: float = 0.5):
        return self._rpc("correlation_matrix", {"method": method, "threshold": threshold})

    def plot_correlation_matrix(self):
        return self._rpc("plot_correlation_matrix", {})

    def compare_columns(self, column1: str, column2: str):
        return self._rpc("compare_columns", {"column1": column1, "column2": column2})

    def get_data_sample(self, n: int = 10, sample_type: str = "head"):
        return self._rpc("get_data_sample", {"n": n, "sample_type": sample_type})

    def group_by_stats(self, group_by: str, column: str):
        return self._rpc("group_by_stats", {"group_by": group_by, "column": column})

    def detect_outliers(self, column: str, method: str = "iqr"):
        return self._rpc("detect_outliers", {"column": column, "method": method})

    def data_quality_report(self):
        return self._rpc("data_quality_report", {})

    def __del__(self):
        self._stop_mcp()
