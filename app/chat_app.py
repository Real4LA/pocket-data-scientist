# app/chat_app.py
# Pocket Data Scientist — Streamlit + Ollama + MCP (upload-first; no backend toggle)

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# --- ensure project root is importable when Streamlit changes CWD ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# -------------------------------------------------------------------

from agents.agent import ToolAwareAgent
from core.tool_client import ToolClient

# Config
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:3b"

PROJECT_ROOT = ROOT
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Pocket Data Scientist", layout="wide")
st.title("🧪 Pocket Data Scientist — Llama3 + MCP")

# ------------------------- Sidebar --------------------------
with st.sidebar:
    st.header("Dataset")
    upload = st.file_uploader(
        "Upload a CSV",
        type=["csv"],
        help="The uploaded file becomes the active dataset for the chat.",
    )

    model = st.text_input("Ollama model", value=DEFAULT_MODEL)

    col_a, col_b = st.columns(2)
    with col_a:
        start = st.button("(Re)connect", type="primary", use_container_width=True)
    with col_b:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------------------- Init state --------------------------
if "tool_client" not in st.session_state or start:
    st.session_state.tool_client = ToolClient()  # MCP stdio client (no direct mode)

if "agent" not in st.session_state or start:
    st.session_state.agent = ToolAwareAgent(
        model=model,
        endpoint=OLLAMA_ENDPOINT,
        tool_client=st.session_state.tool_client,
    )

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = None

# Save uploaded file and set as active dataset on the server via switch_dataset
if upload is not None:
    target = UPLOADS_DIR / upload.name
    with open(target, "wb") as f:
        f.write(upload.getbuffer())

    if target.stat().st_size == 0:
        st.sidebar.error("Uploaded file is empty (0 bytes). Please re-upload.")
    else:
        rel = target.relative_to(PROJECT_ROOT).as_posix()
        try:
            res = st.session_state.tool_client.switch_dataset(rel)
            if isinstance(res, dict):
                if "error" in res:
                    st.sidebar.error(f"Error loading dataset: {res.get('error')}")
                else:
                    st.sidebar.success(f"Loaded {target.name}")
                    st.session_state.active_dataset = res.get('active_dataset', str(target))
            else:
                st.sidebar.warning("Server returned no data for switch_dataset.")
        except Exception as e:
            st.sidebar.error(f"Failed to set dataset: {e}")

# Show active dataset if set
if st.session_state.active_dataset:
    st.caption(f"**Active dataset:** {Path(st.session_state.active_dataset).name}")

# If no dataset selected yet, show a hint
if not st.session_state.active_dataset and upload is None:
    st.info("Upload a CSV in the sidebar, then ask your question (e.g., 'overview', 'numeric summary', 'plot price histogram').")

# --------------------- Chat history -------------------------
for i, m in enumerate(st.session_state.messages):
    # Skip tool messages in main chat - they're only shown in the Tool Usage expander
    if m["role"] == "tool":
        continue
    
    with st.chat_message(m["role"]):
        # Check if there's a plot associated with this assistant message
        # Only show the LAST plot (most recent one)
        if m["role"] == "assistant" and "plot_paths" in m:
            plot_paths = m.get("plot_paths", [])
            # Only display the last plot if there are any
            if plot_paths:
                plot_path = plot_paths[-1]  # Get the last one
                if plot_path:
                    # Path should already be absolute, but ensure it exists
                    plot_file = Path(plot_path)
                    if not plot_file.is_absolute():
                        plot_file = plot_file.resolve()
                    if plot_file.exists():
                        try:
                            st.image(str(plot_file), caption=plot_file.name, width=600)
                        except Exception as e:
                            pass  # Silently fail - don't show warning
        elif m["role"] == "assistant":
            # Fallback: Check previous messages for tool results with plots
            for j in range(i - 1, -1, -1):
                if j < len(st.session_state.messages):
                    prev_msg = st.session_state.messages[j]
                    if prev_msg["role"] == "tool":
                        try:
                            tool_data = json.loads(prev_msg.get("content", "{}"))
                            # Extract path from various structures
                            img_path_str = None
                            if isinstance(tool_data, dict):
                                if "path" in tool_data:
                                    img_path_str = str(tool_data["path"]).strip()
                                elif "structuredContent" in tool_data and isinstance(tool_data["structuredContent"], dict):
                                    if "path" in tool_data["structuredContent"]:
                                        img_path_str = str(tool_data["structuredContent"]["path"]).strip()
                                elif "content" in tool_data and isinstance(tool_data["content"], list) and len(tool_data["content"]) > 0:
                                    first_content = tool_data["content"][0]
                                    if isinstance(first_content, dict) and "text" in first_content:
                                        try:
                                            text_content = json.loads(first_content["text"])
                                            if isinstance(text_content, dict) and "path" in text_content:
                                                img_path_str = str(text_content["path"]).strip()
                                        except (json.JSONDecodeError, TypeError):
                                            pass
                            
                            if img_path_str:
                                    # Try to find the image
                                    img_path = None
                                    # Try as absolute
                                    try:
                                        test_path = Path(img_path_str)
                                        if test_path.is_absolute() and test_path.exists():
                                            img_path = test_path
                                    except:
                                        pass
                                    # Try relative to project root
                                    if not img_path or not img_path.exists():
                                        try:
                                            test_path = PROJECT_ROOT / img_path_str
                                            if test_path.exists():
                                                img_path = test_path
                                        except:
                                            pass
                                    # Try filename in plots directory
                                    if not img_path or not img_path.exists():
                                        try:
                                            filename = Path(img_path_str).name
                                            test_path = PROJECT_ROOT / "plots" / filename
                                            if test_path.exists():
                                                img_path = test_path
                                        except:
                                            pass
                                    # Try resolving
                                    if not img_path or not img_path.exists():
                                        try:
                                            test_path = Path(img_path_str).resolve()
                                            if test_path.exists():
                                                img_path = test_path
                                        except:
                                            pass
                                    # Display if found
                                    if img_path and img_path.exists():
                                        st.image(str(img_path), caption=img_path.name, width=600)
                                        break  # Only show the first plot found
                        except:
                            pass
        
        # User and assistant messages show as markdown
        st.markdown(m.get("content", ""))

# ----------------------- Input box --------------------------
user_input = st.chat_input(
    "Ask me to explore your data… e.g., 'overview', 'numeric summary', 'plot price histogram'."
)

if user_input:
    # Add user message first and display it immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Now process with agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                # Collect all chunks first
                chunks = list(st.session_state.agent.run(st.session_state.messages))
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                import traceback
                st.code(traceback.format_exc())
                chunks = []  # Set empty chunks to avoid errors below  
        
        # Display tool calls (optional, can be collapsed) - OUTSIDE spinner
        try:
            # Debug: check if chunks are empty
            if not chunks:
                st.warning("No chunks received from agent. This might indicate the agent failed to process the request.")
            
            tool_calls = [c for c in chunks if c["type"] == "tool_call"]
            tool_results = [c for c in chunks if c["type"] == "tool_result"]
            assistant_texts = [c for c in chunks if c["type"] == "assistant_text"]
            
            # Debug: show what we got
            if not tool_calls and not assistant_texts:
                st.error(f"Agent returned no tool calls or text. Chunks received: {len(chunks)}")
                if chunks:
                    st.json([c.get("type", "unknown") for c in chunks])
            
            # Match tool calls with their results for display
            tool_call_results_map = {}
            for i, result_chunk in enumerate(tool_results):
                if i < len(tool_calls):
                    tool_call_results_map[i] = {
                        "call": tool_calls[i]["payload"],
                        "result": result_chunk["payload"]
                    }
            
            # Show tool calls with their outputs in an expander
            if tool_calls:
                with st.expander("🔧 Tool Usage", expanded=False):
                    for i, chunk in enumerate(tool_calls):
                        call_payload = chunk["payload"]
                        st.subheader(f"Tool: {call_payload.get('name', 'unknown')}")
                        
                        # Show tool call with args
                        st.write("**Call:**")
                        st.code(json.dumps(call_payload, indent=2), language="json")
                        
                        # Show corresponding result if available
                        if i in tool_call_results_map:
                            result_payload = tool_call_results_map[i]["result"]
                            st.write("**Output:**")
                            
                            # If it's a plot, show the image in the expander too
                            # Extract path from various structures
                            plot_path = None
                            if isinstance(result_payload, dict):
                                if "path" in result_payload:
                                    plot_path = result_payload["path"]
                                elif "structuredContent" in result_payload and isinstance(result_payload["structuredContent"], dict):
                                    plot_path = result_payload["structuredContent"].get("path")
                                elif "content" in result_payload and isinstance(result_payload["content"], list) and len(result_payload["content"]) > 0:
                                    first_content = result_payload["content"][0]
                                    if isinstance(first_content, dict) and "text" in first_content:
                                        try:
                                            text_content = json.loads(first_content["text"])
                                            if isinstance(text_content, dict) and "path" in text_content:
                                                plot_path = text_content["path"]
                                        except (json.JSONDecodeError, TypeError):
                                            pass
                            
                            if plot_path:
                                img_path = Path(plot_path)
                                if not img_path.is_absolute():
                                    img_path = PROJECT_ROOT / img_path
                                if img_path.exists():
                                    st.image(str(img_path), caption=img_path.name, width=600)
                            
                            # Show JSON output
                            st.code(json.dumps(result_payload, indent=2), language="json")
                        
                        if i < len(tool_calls) - 1:
                            st.divider()
            
            # Show tool results in main chat (especially plots) - plots MUST appear here BEFORE assistant text
            # Only collect the LAST plot path (most recent plot generated)
            last_plot_path = None
            for chunk in tool_results:
                payload = chunk["payload"]
                # Preview image if present - show prominently in chat BEFORE assistant text
                # Extract path from various possible structures
                img_path_str = None
                if isinstance(payload, dict):
                    # Try direct path first
                    if "path" in payload and payload.get("path"):
                        img_path_str = str(payload["path"]).strip()
                    # Try structuredContent.path (FastMCP format)
                    elif "structuredContent" in payload and isinstance(payload["structuredContent"], dict):
                        if "path" in payload["structuredContent"]:
                            img_path_str = str(payload["structuredContent"]["path"]).strip()
                    # Try content[0].text (JSON string format)
                    elif "content" in payload and isinstance(payload["content"], list) and len(payload["content"]) > 0:
                        first_content = payload["content"][0]
                        if isinstance(first_content, dict) and "text" in first_content:
                            try:
                                # Try to parse as JSON
                                text_content = json.loads(first_content["text"])
                                if isinstance(text_content, dict) and "path" in text_content:
                                    img_path_str = str(text_content["path"]).strip()
                            except (json.JSONDecodeError, TypeError):
                                # If not JSON, check if it's a direct path string
                                text_val = first_content["text"]
                                # Try to extract path from text (handles both .png and other image formats)
                                import re
                                # Look for paths in quotes
                                match = re.search(r'["\']([^"\']*\.(png|jpg|jpeg))["\']', text_val, re.IGNORECASE)
                                if match:
                                    img_path_str = match.group(1).strip()
                                # Also try extracting from JSON-like strings
                                elif "path" in text_val.lower():
                                    # Try to find path value after "path"
                                    path_match = re.search(r'["\']path["\']\s*:\s*["\']([^"\']+)["\']', text_val, re.IGNORECASE)
                                    if path_match:
                                        img_path_str = path_match.group(1).strip()
                
                if img_path_str:
                    # Try multiple path resolution strategies
                    img_path = None
                    
                    # Strategy 1: Try relative to project root (most common case - paths like "plots/filename.png")
                    try:
                        test_path = PROJECT_ROOT / img_path_str
                        if test_path.exists():
                            img_path = test_path.resolve()
                    except Exception as e:
                        pass
                    
                    # Strategy 2: Try just the filename in plots directory
                    if not img_path or not img_path.exists():
                        try:
                            filename = Path(img_path_str).name
                            # Try exact filename match
                            test_path = PROJECT_ROOT / "plots" / filename
                            if test_path.exists():
                                img_path = test_path.resolve()
                            # Also try partial match for correlation_matrix files
                            elif "correlation" in filename.lower() or "correlation_matrix" in img_path_str.lower():
                                # Find the most recent correlation matrix file (by modification time)
                                plots_dir = PROJECT_ROOT / "plots"
                                if plots_dir.exists():
                                    correlation_files = list(plots_dir.glob("correlation_matrix*.png"))
                                    if correlation_files:
                                        # Sort by modification time, get most recent
                                        correlation_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                                        img_path = correlation_files[0].resolve()
                        except Exception as e:
                            pass
                    
                    # Strategy 3: Try as absolute path
                    if not img_path or not img_path.exists():
                        try:
                            test_path = Path(img_path_str)
                            if test_path.is_absolute() and test_path.exists():
                                img_path = test_path.resolve()
                        except Exception as e:
                            pass
                    
                    # Strategy 4: Try resolving the path
                    if not img_path or not img_path.exists():
                        try:
                            test_path = Path(img_path_str).resolve()
                            if test_path.exists():
                                img_path = test_path
                        except Exception as e:
                            pass
                    
                    # Store the path (will be overwritten by later plots, keeping only the last one)
                    if img_path and img_path.exists():
                        last_plot_path = str(img_path)
            
            # Display only the LAST plot (most recent one)
            if last_plot_path:
                try:
                    st.image(last_plot_path, caption=Path(last_plot_path).name, width=600)
                except Exception as e:
                    pass  # Silently fail
            
            # Store tool results for history (for Tool Usage expander)
            for chunk in tool_results:
                payload = chunk["payload"]
                # Store result (but don't show JSON in main chat, only in expander)
                if isinstance(payload, dict) and "error" in payload:
                    st.warning(f"Error: {payload.get('error')}")
                # Store tool results for history (for Tool Usage expander)
                st.session_state.messages.append(
                    {"role": "tool", "content": json.dumps(payload)}
                )
            
            # Store tool calls in session state after displaying
            for chunk in tool_calls:
                st.session_state.messages.append(
                    {"role": "tool", "content": json.dumps(chunk["payload"])}
                )
                
            # Display assistant's natural language response (this is the main output)
            if assistant_texts:
                for chunk in assistant_texts:
                    st.markdown(chunk["content"])
                    # Store assistant message with only the last plot path for persistence
                    assistant_msg = {
                        "role": "assistant",
                        "content": chunk["content"],
                        "plot_paths": [last_plot_path] if last_plot_path else []  # Store only the last plot
                    }
                    st.session_state.messages.append(assistant_msg)
            else:
                # Fallback if no text response
                st.info("Received tool results but no text response generated.")
        except Exception as e:
            st.error(f"Unexpected error processing response: {e}")
            import traceback
            st.code(traceback.format_exc())
