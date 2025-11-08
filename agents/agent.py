import json
import re
import requests
from typing import Dict, Iterable, Any

# Optimized system prompt for tool selection
SYSTEM_DECIDE = (
    "You are a data analyst assistant. Help with data analysis questions about the dataset.\n\n"
    "AVAILABLE TOOLS:\n"
    "Data exploration:\n"
    "- list_columns(): Get column names and types (REQUIRED before tools needing column names)\n"
    "- get_data_overview(): Dataset overview\n"
    "- summary_stats(): Statistics for all numeric columns\n"
    "\n"
    "Column-specific analysis (REQUIRES list_columns() first):\n"
    "- get_column_summary(column='Name'): Column details and top values\n"
    "- group_by_stats(group_by='Category', column='Numeric'): Statistics grouped by category\n"
    "- plot_column(column='X', kind='scatter', y='Y'): Scatter plot for X vs Y\n"
    "- plot_column(column='X', kind='bar'): Bar chart for category counts\n"
    "- plot_column(column='X', kind='hist'): Histogram for distributions\n"
    "\n"
    "Correlation analysis:\n"
    "- correlation_matrix(): Correlation data (text only)\n"
    "- plot_correlation_matrix(): Correlation heatmap (all correlations)\n"
    "\n"
    "TOOL SELECTION RULES:\n"
    "1. Column name tools: get_column_summary(), plot_column(), group_by_stats() MUST be preceded by list_columns()\n"
    "2. Use EXACT column names from list_columns() result (case-sensitive)\n"
    "\n"
    "EXAMPLES OF PROPER TOOL USAGE:\n"
    "\n"
    "Example 1: User asks 'what is the highest paying job?'\n"
    "Step 1: {\"tool_call\":{\"name\":\"list_columns\",\"args\":{}}}\n"
    "Step 2: After seeing columns like 'Job_Title' and 'Salary_USD', call:\n"
    "        {\"tool_call\":{\"name\":\"group_by_stats\",\"args\":{\"group_by\":\"Job_Title\",\"column\":\"Salary_USD\"}}}\n"
    "\n"
    "Example 2: User asks 'visualize the relationship between age and salary'\n"
    "Step 1: {\"tool_call\":{\"name\":\"list_columns\",\"args\":{}}}\n"
    "Step 2: After seeing columns like 'Age' and 'Salary_USD', call:\n"
    "        {\"tool_call\":{\"name\":\"plot_column\",\"args\":{\"column\":\"Age\",\"kind\":\"scatter\",\"y\":\"Salary_USD\"}}}\n"
    "\n"
    "Example 3: User asks 'plot correlation matrix'\n"
    "Direct call: {\"tool_call\":{\"name\":\"plot_correlation_matrix\",\"args\":{}}}\n"
    "(No list_columns() needed - this tool works on all numeric columns)\n"
    "\n"
    "Example 4: User asks 'what are the most common cities?'\n"
    "Step 1: {\"tool_call\":{\"name\":\"list_columns\",\"args\":{}}}\n"
    "Step 2: After seeing column 'City', call:\n"
    "        {\"tool_call\":{\"name\":\"get_column_summary\",\"args\":{\"column\":\"City\"}}}\n"
    "\n"
    "Example 5: User asks 'show me summary statistics'\n"
    "Direct call: {\"tool_call\":{\"name\":\"summary_stats\",\"args\":{}}}\n"
    "(No list_columns() needed)\n"
    "\n"
    "Question → Tool mapping:\n"
    "- 'dataset overview'/'what is this data' → get_data_overview()\n"
    "- 'summary statistics' → summary_stats()\n"
    "- 'most common X'/'top values in X' → list_columns() then get_column_summary(column='X')\n"
    "- 'average Y by X'/'highest Y by X' → list_columns() then group_by_stats(group_by='X', column='Y')\n"
    "- 'plot correlation matrix'/'visualize correlation matrix' → plot_correlation_matrix()\n"
    "- 'correlation' (no plot) → correlation_matrix()\n"
    "- 'X vs Y'/'relationship between X and Y' with 'plot'/'visualize' → list_columns() then plot_column(column='X', kind='scatter', y='Y')\n"
    "- 'plot X'/'visualize X' (single variable) → list_columns() then plot_column(column='X', kind='auto')\n"
    "\n"
    "CRITICAL: 'plot correlation matrix' = heatmap (plot_correlation_matrix). 'plot X vs Y' = scatter plot (plot_column). Never use correlation_matrix() when user asks to plot/visualize.\n"
    "\n"
    "For non-data questions (greetings, general chat), return null immediately.\n"
    "Output: {\"tool_call\":{\"name\":\"tool_name\",\"args\":{...}}} or null"
)

SYSTEM_CHAT = (
    "You are a helpful data analyst assistant.\n\n"
    "Response guidelines:\n"
    "- Use EXACT values from tool results (no rounding, estimating, or inferring)\n"
    "- Answer ONLY what was asked - stay relevant\n"
    "- If data isn't in tool results, you don't have that information\n"
    "- For plots: Simply confirm 'Plot created successfully'\n"
    "- Format numbers clearly (e.g., $149,724)\n"
    "- Use plain text only - no markdown\n"
    "- Be concise and natural\n\n"
    "For general conversation: Be friendly and helpful. Explain you can analyze datasets."
)

TOOLCALL_RE = re.compile(r'\{\s*"tool_call"\s*:\s*\{[\s\S]*?\}\s*\}')

def _extract_tool_call(text: str):
    if not text or not text.strip():
        return None
    
    # Try parsing as JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool_call" in obj:
            return obj["tool_call"]
        # Sometimes the tool_call is at the root level
        if isinstance(obj, dict) and "name" in obj and "args" in obj:
            return obj
    except Exception:
        pass
    
    # Try regex extraction
    m = TOOLCALL_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "tool_call" in obj:
                return obj["tool_call"]
        except Exception:
            pass
    
    # Try to find JSON-like structure even if not perfect
    try:
        # Look for patterns like {"name": "...", "args": {...}}
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
        if name_match:
            name = name_match.group(1)
            # Try to extract args
            args_match = re.search(r'"args"\s*:\s*(\{[^}]*\})', text)
            args = {}
            if args_match:
                try:
                    args = json.loads(args_match.group(1))
                except:
                    pass
            return {"name": name, "args": args}
    except Exception:
        pass
    
    return None

class ToolAwareAgent:
    def __init__(self, model: str, endpoint: str, tool_client):
        self.model = model
        self.endpoint = endpoint
        self.tool_client = tool_client

    def _chat(self, messages: list[dict], *, format_json: bool = False, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if format_json:
            payload["format"] = "json"  # Ollama JSON-only response
        r = requests.post(self.endpoint, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _decide_tool(self, history: list[dict]) -> dict | None:
        try:
            msgs = [{"role": "system", "content": SYSTEM_DECIDE}] + history
            text = self._chat(msgs, format_json=True, temperature=0.0)
            
            # Check if LLM returned null (no tool call needed)
            if text.strip().lower() in ['null', 'none', '']:
                return None
            
            call = _extract_tool_call(text)
            if call and isinstance(call, dict) and "name" in call:
                # Reject if name is "null" - this means no tool call
                if call.get("name", "").lower() in ['null', 'none']:
                    return None
                return call
            # Retry with more explicit instruction
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "system", "content": "You must output ONLY a valid JSON object with tool_call. Example: {\"tool_call\":{\"name\":\"plot_column\",\"args\":{\"column\":\"ColumnName\"}}}. If no tool is needed, output just: null"})
            text = self._chat(msgs, format_json=True, temperature=0.0)
            
            # Check if LLM returned null
            if text.strip().lower() in ['null', 'none', '']:
                return None
            
            call = _extract_tool_call(text)
            if call and isinstance(call, dict) and "name" in call:
                # Reject if name is "null"
                if call.get("name", "").lower() in ['null', 'none']:
                    return None
                return call
        except Exception as e:
            # Log error but don't crash
            print(f"Error in _decide_tool: {e}")
        return None

    def _execute_tool(self, name: str, args: dict) -> dict:
        """Execute a single tool and return result."""
        try:
            if name == "get_data_overview":
                result = self.tool_client.get_data_overview()
            elif name == "list_columns":
                result = self.tool_client.list_columns()
            elif name == "summary_stats":
                result = self.tool_client.summary_stats()
            elif name == "get_column_summary":
                result = self.tool_client.get_column_summary(args.get("column", ""))
            elif name == "plot_column":
                result = self.tool_client.plot_column(
                    column=args.get("column", ""),
                    kind=args.get("kind", "auto"),
                    y=args.get("y"),
                    group_by=args.get("group_by"),
                    bins=args.get("bins", 30),
                )
            elif name == "correlation_matrix":
                result = self.tool_client.correlation_matrix(
                    method=args.get("method", "pearson"),
                    threshold=args.get("threshold", 0.5),
                )
            elif name == "plot_correlation_matrix":
                result = self.tool_client.plot_correlation_matrix()
            elif name == "compare_columns":
                result = self.tool_client.compare_columns(
                    column1=args.get("column1", ""),
                    column2=args.get("column2", ""),
                )
            elif name == "get_data_sample":
                result = self.tool_client.get_data_sample(
                    n=args.get("n", 10),
                    sample_type=args.get("sample_type", "head"),
                )
            elif name == "group_by_stats":
                result = self.tool_client.group_by_stats(
                    group_by=args.get("group_by", ""),
                    column=args.get("column", ""),
                )
            elif name == "detect_outliers":
                result = self.tool_client.detect_outliers(
                    column=args.get("column", ""),
                    method=args.get("method", "iqr"),
                )
            elif name == "data_quality_report":
                result = self.tool_client.data_quality_report()
            else:
                result = {"error": f"Unknown tool '{name}'"}
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            result = {"error": f"Tool execution failed: {str(e)}", "traceback": error_trace}
        
        if result is None:
            result = {"error": "Tool returned None"}
        
        return result

    def run(self, history: list[dict]) -> Iterable[Dict]:
        """Run agent with support for multi-tool chaining."""
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        all_tool_results = []
        current_history = history.copy()
        called_tools = set()  # Track called tools to prevent duplicates
        
        while iteration < max_iterations:
            iteration += 1
            
            # 1) Ask for next action
            call = self._decide_tool(current_history)
            
            if not call:
                # No more tools to call, generate final answer
                break
            
            # 2) Execute tool
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            
            # Tools that require column names - must have list_columns() called first
            tools_requiring_columns = ["get_column_summary", "plot_column", "group_by_stats"]
            
            # Check if we need to call list_columns() first
            if name in tools_requiring_columns:
                # Check if list_columns() has been called
                list_columns_called = any(tr["tool"] == "list_columns" for tr in all_tool_results)
                
                if not list_columns_called:
                    # Force call to list_columns() first
                    list_columns_key = ("list_columns", "{}")
                    if list_columns_key not in called_tools:
                        # Call list_columns() first
                        yield {"type": "tool_call", "payload": {"name": "list_columns", "args": {}}}
                        list_result = self._execute_tool("list_columns", {})
                        extracted_list_result = self._extract_actual_data(list_result)
                        yield {"type": "tool_result", "payload": extracted_list_result}
                        all_tool_results.append({"tool": "list_columns", "result": extracted_list_result})
                        called_tools.add(list_columns_key)
                        
                        # Add to history
                        list_summary = self._summarize_tool_result("list_columns", extracted_list_result)
                        current_history.append({
                            "role": "assistant",
                            "content": f"Used tool list_columns",
                        })
                        current_history.append({
                            "role": "user",
                            "content": f"Tool result: {list_summary}\n\nColumn names retrieved. Now use these EXACT column names (case-sensitive) to call the tool you were about to call: {name}. Do NOT use lowercase or guessed column names - use the exact names from the list above."
                        })
                        # Continue to next iteration to let LLM decide again with column names available
                        continue
            
            # Create a unique key for this tool call to prevent duplicates
            tool_key = (name, json.dumps(args, sort_keys=True))
            
            # Prevent calling the same tool with same args multiple times
            if tool_key in called_tools:
                # Already called this tool with these args, skip and generate answer
                break
            
            called_tools.add(tool_key)
            yield {"type": "tool_call", "payload": call}
            
            if not name:
                result = {"error": "Tool name is missing from tool call"}
                yield {"type": "tool_result", "payload": result}
                all_tool_results.append({"tool": name, "result": result})
                break
            
            # Execute tool
            result = self._execute_tool(name, args)
            
            # Extract actual data from nested structures
            extracted_result = self._extract_actual_data(result)
            yield {"type": "tool_result", "payload": extracted_result}
            all_tool_results.append({"tool": name, "result": extracted_result})
            
            # Add tool result to history for next iteration
            # Format it clearly so LLM can see if more tools are needed
            summary = self._summarize_tool_result(name, extracted_result)
            current_history.append({
                "role": "assistant",
                "content": f"Used tool {name}",
            })
            original_question = history[-1].get('content', 'Analyze the data') if history else 'Analyze the data'
            
            # Check if result indicates we need more data (0 rows, empty, error)
            needs_more = False
            if "error" in extracted_result:
                needs_more = True
            elif isinstance(extracted_result, dict):
                # Check for empty data indicators
                shape = extracted_result.get("shape", {})
                if isinstance(shape, dict) and shape.get("rows", 1) == 0:
                    needs_more = True
                elif extracted_result.get("total_rows", 1) == 0:
                    needs_more = True
                elif not extracted_result or len(extracted_result) == 0:
                    needs_more = True
            
            # Build optimized context message
            question_lower = original_question.lower()
            asked_for_plot = any(word in question_lower for word in ['plot', 'visualize', 'show', 'graph', 'chart'])
            asked_for_relationship_plot = any(phrase in question_lower for phrase in ['vs', 'relationship', 'relation']) and asked_for_plot
            asked_for_correlation_matrix_plot = 'correlation matrix' in question_lower and asked_for_plot
            
            context_msg = f"Tool result: {summary}\n\nQuestion: {original_question}\n\n"
            
            if needs_more:
                if asked_for_correlation_matrix_plot and name != "plot_correlation_matrix":
                    context_msg += "User asked to PLOT correlation matrix. Call plot_correlation_matrix(), not correlation_matrix()."
                else:
                    context_msg += "Data incomplete. Call get_data_overview() or list_columns() to verify."
            elif name in ["plot_column", "plot_correlation_matrix"]:
                plots_created = [tr for tr in all_tool_results if tr['tool'] in ['plot_column', 'plot_correlation_matrix']]
                plot_keywords = ['scatter', 'histogram', 'bar chart', 'correlation matrix', 'visualize', 'plot']
                plot_requests_count = sum(1 for keyword in plot_keywords if keyword in question_lower)
                if plot_requests_count > len(plots_created):
                    context_msg += f"Plot created. {plot_requests_count - len(plots_created)} more plot(s) needed. Continue."
                else:
                    context_msg += "Plot created. Return null."
            elif name == "correlation_matrix":
                if asked_for_correlation_matrix_plot:
                    context_msg += "User asked to PLOT correlation matrix. Call plot_correlation_matrix() instead. Do not return null."
                elif asked_for_relationship_plot:
                    context_msg += "User asked to VISUALIZE relationship between two variables. Call plot_column(column='X', kind='scatter', y='Y'). Do not return null."
                else:
                    context_msg += "Correlation data retrieved. Return null."
            elif name == "list_columns":
                if asked_for_relationship_plot:
                    context_msg += "Use EXACT column names (case-sensitive) from above to call plot_column(column='X', kind='scatter', y='Y'). Do not return null."
                else:
                    context_msg += "Use EXACT column names (case-sensitive) from above to call the appropriate tool. Return null only after calling needed tools."
            elif name in ["group_by_stats", "get_column_summary"]:
                context_msg += "Data retrieved. Return null."
            elif name == "get_data_overview":
                context_msg += "Overview retrieved. Review question - call more tools if needed. Return null only if you have all data."
            else:
                context_msg += "Review question - call more tools if needed. Return null only if you have all data."
            
            current_history.append({
                "role": "user",
                "content": context_msg
            })
        
        # 3) Generate final answer based on all tool results
        # If no tools were called, generate a natural conversational response
        if not all_tool_results:
            # No tools called - this is likely a general conversation question
            # Generate a friendly, conversational response
            original_question = history[-1].get('content', '') if history else ''
            followup = history + [
                {"role": "user", "content": f"User said: {original_question}\n\nRespond naturally and conversationally. Be friendly and helpful. If they're asking what you can do, explain that you can help analyze datasets."}
            ]
            msgs = [{"role": "system", "content": SYSTEM_CHAT}] + followup
            final_text = self._chat(msgs, temperature=0.7)  # Higher temperature for more natural conversation
            yield {"type": "assistant_text", "content": final_text}
            return
        
        if all_tool_results:
            # Combine all tool results into a comprehensive summary
            combined_summary = "DATA ANALYSIS RESULTS FROM ALL TOOLS:\n\n"
            for i, tr in enumerate(all_tool_results):
                tool_name = tr["tool"]
                tool_result = tr["result"]
                summary = self._summarize_tool_result(tool_name, tool_result)
                combined_summary += f"Tool {i+1}: {tool_name}\n{summary}\n\n"
            
            combined_data = "\n\n".join([
                f"Tool: {tr['tool']}\n{json.dumps(tr['result'], indent=2)}"
                for tr in all_tool_results
            ])
            
            # Check if the original question was ONLY asking for a plot/visualization
            original_question_lower = history[-1].get('content', '').lower() if history else ''
            is_plot_only_request = any(phrase in original_question_lower for phrase in [
                'plot', 'visualize', 'show chart', 'create graph', 'draw', 'graph', 'chart'
            ]) and not any(phrase in original_question_lower for phrase in [
                'analyze', 'what', 'how many', 'tell me', 'explain', 'describe', 'summary', 'overview', 'insights'
            ])
            
            # Check if a plot was actually created
            plot_created = any(tr['tool'] in ['plot_column', 'plot_correlation_matrix'] for tr in all_tool_results)
            
            # Special handling for correlation matrix plots - they should show the plot even if it's a plot-only request
            correlation_plot_created = any(tr['tool'] == 'plot_correlation_matrix' for tr in all_tool_results)
            
            if is_plot_only_request and plot_created and not correlation_plot_created:
                tool_result_text = "Plot created. Respond: 'Plot created successfully.'"
            elif correlation_plot_created:
                tool_result_text = f"""{combined_summary}

DATA: {combined_data}

Correlation matrix plot created. Answer concisely - mention plot created and top correlations if available."""
            else:
                tool_result_text = f"""{combined_summary}

DATA: {combined_data}

Answer the question using exact values from tool results. Be concise and relevant."""
            
            followup = history + [
                {"role": "assistant", "content": "Analyzing the data with multiple tools..."},
                {"role": "user", "content": tool_result_text},
            ]
            answer = self._chat([{"role": "system", "content": SYSTEM_CHAT}] + followup, temperature=0.2)
            yield {"type": "assistant_text", "content": answer}
        else:
            # Fallback if no tools were called
            answer = self._chat([{"role": "system", "content": SYSTEM_CHAT}] + history, temperature=0.2)
            yield {"type": "assistant_text", "content": answer}
    
    def _extract_actual_data(self, result: Any) -> dict:
        """Extract actual data from nested FastMCP response structures."""
        # Handle None or non-dict types
        if result is None:
            return {"error": "Tool returned None"}
        
        if not isinstance(result, dict):
            # If it's a list or other type, wrap it
            return {"data": result}
        
        # FastMCP might wrap results in different structures
        # Try structuredContent first (common FastMCP format)
        if "structuredContent" in result:
            content = result["structuredContent"]
            if isinstance(content, dict):
                return self._extract_actual_data(content)  # Recursive extraction
            return {"data": content}
        
        # Try content array (MCP protocol format)
        if "content" in result and isinstance(result["content"], list):
            for item in result["content"]:
                if isinstance(item, dict):
                    # Try text field with JSON
                    if "text" in item:
                        try:
                            parsed = json.loads(item["text"])
                            if isinstance(parsed, dict):
                                return parsed
                        except (json.JSONDecodeError, TypeError):
                            pass
                    # Try direct dict content
                    if isinstance(item, dict) and len(item) > 0:
                        # Check if it looks like actual data (has keys like 'shape', 'columns', 'stats', etc.)
                        if any(key in item for key in ['shape', 'columns', 'stats', 'summary', 'path', 'error']):
                            return item
        
        # Try result field (some MCP implementations)
        if "result" in result and isinstance(result["result"], dict):
            return result["result"]
        
        # Check if this already looks like actual data (has expected keys)
        # If it has data-like keys, it's probably already the actual data
        data_keys = ['shape', 'columns', 'stats', 'summary', 'path', 'error', 'total_rows', 
                     'numeric_columns', 'text_columns', 'date_columns', 'dtypes', 'memory_usage_mb']
        if any(key in result for key in data_keys):
            return result
        
        # If no nested structure found, return as-is (might already be the data)
        return result
    
    def _summarize_tool_result(self, tool_name: str, result: dict) -> str:
        """Create a rich, human-readable summary of tool results with insights."""
        # Extract actual data from nested structures
        result = self._extract_actual_data(result)
        
        # Ensure result is a dict
        if not isinstance(result, dict):
            return f"Unexpected result type: {type(result)}"
        
        if "error" in result:
            return f"ERROR: {result.get('error')}"
        
        if tool_name == "get_data_overview":
            overview = result
            # Safely extract shape
            shape = overview.get('shape', {})
            if isinstance(shape, dict):
                rows = shape.get('rows', 0)
                cols = shape.get('columns', 0)
            else:
                # Fallback if shape is not a dict
                rows = overview.get('total_rows', 0)
                cols = len(overview.get('columns', []))
            
            text = f"Dataset Overview:\n"
            text += f"  • Shape: {rows:,} rows × {cols} columns\n"
            text += f"  • Memory usage: {overview.get('memory_usage_mb', 0):.2f} MB\n"
            text += f"  • Duplicate rows: {overview.get('duplicate_rows', 0):,}\n"
            numeric_cols = overview.get('numeric_columns', [])
            categorical_cols = overview.get('categorical_columns', [])
            text += f"  • Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'None'}\n"
            text += f"  • Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'None'}\n"
            if overview.get('date_columns'):
                text += f"  • Date columns: {', '.join(overview.get('date_columns', []))}\n"
            if overview.get('columns'):
                text += f"  • All columns: {', '.join(overview.get('columns', []))}\n"
            return text
        
        elif tool_name == "list_columns":
            # Handle both dict and list formats for columns
            cols = result.get("columns", {})
            if isinstance(cols, list):
                # If columns is a list, convert to dict with dtypes if available
                dtypes = result.get("dtypes", {})
                cols = {col: dtypes.get(col, "unknown") for col in cols} if isinstance(dtypes, dict) else {col: "unknown" for col in cols}
            
            total_rows = result.get("total_rows", 0)
            numeric_cols = result.get("numeric_columns", [])
            text_cols = result.get("text_columns", [])
            date_cols = result.get("date_columns", [])
            
            summary = f"Dataset Columns:\n"
            summary += f"  • Total rows: {total_rows:,}\n"
            summary += f"  • Total columns: {len(cols) if cols else 0}\n"
            if numeric_cols:
                summary += f"  • Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols)}\n"
            if text_cols:
                summary += f"  • Text columns ({len(text_cols)}): {', '.join(text_cols)}\n"
            if date_cols:
                summary += f"  • Date columns ({len(date_cols)}): {', '.join(date_cols)}\n"
            summary += f"\nColumn Details:\n"
            if cols:
                for col, dtype in cols.items():
                    summary += f"  • {col}: {dtype}\n"
            else:
                summary += "  No columns found in result.\n"
            return summary
        
        elif tool_name == "summary_stats":
            stats = result.get("stats", {})
            insights = result.get("insights", [])
            total_records = result.get("total_records", 0)
            
            if stats:
                summary = f"Statistical Summary ({len(stats)} numeric column(s), {total_records:,} records):\n\n"
                summary += "⚠️ CRITICAL: Use ONLY the exact values shown below. DO NOT round, estimate, or change these numbers.\n\n"
                for col, values in stats.items():
                    count = values.get("count", "N/A")
                    mean = values.get("mean", "N/A")
                    std = values.get("std", "N/A")
                    min_val = values.get("min", "N/A")
                    max_val = values.get("max", "N/A")
                    
                    summary += f"{col}:\n"
                    summary += f"  • Count: {count:,}\n"
                    # Format numbers properly - preserve exact values, show both formatted and raw
                    if isinstance(mean, (int, float)):
                        summary += f"  • Mean: {mean:,.2f} (exact: {mean})\n"
                    else:
                        summary += f"  • Mean: {mean}\n"
                    if isinstance(std, (int, float)):
                        summary += f"  • Std: {std:,.2f} (exact: {std})\n"
                    else:
                        summary += f"  • Std: {std}\n"
                    if isinstance(min_val, (int, float)):
                        summary += f"  • Min: {min_val:,.2f} (EXACT: {min_val}) ⚠️ USE THIS EXACT NUMBER\n"
                    else:
                        summary += f"  • Min: {min_val} ⚠️ USE THIS EXACT NUMBER\n"
                    if isinstance(max_val, (int, float)):
                        summary += f"  • Max: {max_val:,.2f} (EXACT: {max_val}) ⚠️ USE THIS EXACT NUMBER - DO NOT USE ANY OTHER VALUE\n"
                    else:
                        summary += f"  • Max: {max_val} ⚠️ USE THIS EXACT NUMBER - DO NOT USE ANY OTHER VALUE\n"
                    summary += "\n"
                
                if insights:
                    summary += "Insights:\n"
                    for insight in insights[:3]:  # Show top 3 insights
                        summary += f"  • {insight}\n"
                return summary
            return "No statistics available."
        
        elif tool_name == "get_column_summary":
            summary = result.get("summary", {})
            if summary:
                summary_text = f"Column Summary:\n"
                if "total_rows" in summary:
                    summary_text += f"  • Total rows: {summary['total_rows']:,}\n"
                if "null_percentage" in summary:
                    summary_text += f"  • Null percentage: {summary['null_percentage']:.1f}%\n"
                if "unique_percentage" in summary:
                    summary_text += f"  • Unique values: {summary['unique_percentage']:.1f}%\n"
                
                if "numeric" in summary:
                    num = summary["numeric"]
                    summary_text += f"\nNumeric Statistics:\n"
                    summary_text += f"  • Mean: {num.get('mean', 'N/A'):,.2f}\n"
                    summary_text += f"  • Median: {num.get('median', 'N/A'):,.2f}\n"
                    summary_text += f"  • Range: {num.get('min', 'N/A'):,.2f} to {num.get('max', 'N/A'):,.2f}\n"
                    summary_text += f"  • IQR: {num.get('iqr', 'N/A'):,.2f}\n"
                
                if "top_values" in summary:
                    summary_text += f"\n⚠️ Most Common Values (EXACT COUNTS - use these exact numbers):\n"
                    for item in summary["top_values"][:5]:
                        exact_count = item['count']
                        summary_text += f"  • {item['value']}: {exact_count} times (EXACT COUNT: {exact_count}) ({item.get('percentage', 0):.1f}%)\n"
                    summary_text += f"⚠️ CRITICAL: Use ONLY these exact counts. If it shows Denver: 11, use 11 (not 6, not any other number).\n"
                
                if "insights" in summary:
                    insights = summary["insights"]
                    summary_text += f"\nInsights:\n"
                    summary_text += f"  • Distribution: {insights.get('distribution', insights.get('type', 'N/A'))}\n"
                    summary_text += f"  • Note: {insights.get('note', 'N/A')}\n"
                
                return summary_text
            return "No summary available."
        
        elif tool_name == "plot_column":
            # Check for errors first
            if "error" in result:
                error_msg = result.get("error", "Unknown error")
                column = result.get("column", "unknown")
                kind = result.get("kind", "unknown")
                y = result.get("y")
                if y:
                    return f"❌ Plot generation failed for {column} vs {y} ({kind}): {error_msg}"
                else:
                    return f"❌ Plot generation failed for {column} ({kind}): {error_msg}"
            
            # Extract path from various possible structures
            path = result.get("path")
            if not path and "structuredContent" in result:
                path = result.get("structuredContent", {}).get("path")
            if not path and "content" in result:
                try:
                    content = result.get("content", [])
                    if content and isinstance(content[0], dict) and "text" in content[0]:
                        text_data = json.loads(content[0]["text"])
                        path = text_data.get("path")
                except:
                    pass
            
            if path:
                column = result.get("column", "the column")
                kind = result.get("kind", "visualization")
                y = result.get("y")
                if y and kind == "scatter":
                    return f"✅ Scatter plot created successfully!\n  • Saved to: {path}\n  • Plot type: {kind}\n  • X-axis: {column}\n  • Y-axis: {y}"
                else:
                    return f"✅ Visualization created successfully!\n  • Saved to: {path}\n  • Plot type: {kind}\n  • Column: {column}"
            return "❌ Plot generation failed: No path returned from tool."
        
        elif tool_name == "correlation_matrix":
            if "error" in result:
                return f"Error: {result.get('error')}"
            
            matrix = result.get("matrix", {})
            if matrix:
                cols = result.get("columns", [])
                method = result.get("method", "pearson")
                strong_corrs = result.get("strong_correlations", [])
                
                text = f"Correlation Analysis ({method} method):\n"
                text += f"  • Analyzed {len(cols)} numeric columns: {', '.join(cols)}\n\n"
                
                # Show all correlations sorted by absolute value
                all_corrs = []
                for i, col1 in enumerate(cols):
                    for j, col2 in enumerate(cols):
                        if i != j and col1 in matrix and col2 in matrix[col1]:
                            corr_val = matrix[col1][col2]
                            all_corrs.append({
                                "var1": col1,
                                "var2": col2,
                                "value": corr_val
                            })
                
                # Sort by absolute correlation value
                all_corrs.sort(key=lambda x: abs(x["value"]), reverse=True)
                
                if all_corrs:
                    text += "Top Correlations:\n"
                    for corr in all_corrs[:10]:  # Show top 10
                        strength = "strong" if abs(corr["value"]) >= 0.7 else "moderate" if abs(corr["value"]) >= 0.5 else "weak"
                        text += f"  • {corr['var1']} vs {corr['var2']}: {corr['value']:.3f} ({strength})\n"
                
                if strong_corrs:
                    text += f"\nStrong correlations (threshold: {result.get('threshold', 0.5)}):\n"
                    for corr in strong_corrs[:5]:
                        text += f"  • {corr['variable1']} vs {corr['variable2']}: {corr['correlation']:.3f} ({corr['strength']})\n"
                
                return text
            return result.get("error", "No correlation data available.")
        
        elif tool_name == "plot_correlation_matrix":
            # Extract path from various possible structures
            path = result.get("path")
            if not path and "structuredContent" in result:
                path = result.get("structuredContent", {}).get("path")
            if not path and "content" in result:
                try:
                    content = result.get("content", [])
                    if content and isinstance(content[0], dict) and "text" in content[0]:
                        text_data = json.loads(content[0]["text"])
                        path = text_data.get("path")
                except:
                    pass
            
            if path:
                columns = result.get("columns", [])
                matrix = result.get("matrix", {})
                # Extract non-diagonal correlations only (ignore self-correlations)
                non_diagonal_corrs = []
                if matrix and columns:
                    for i, col1 in enumerate(columns):
                        for j, col2 in enumerate(columns):
                            if i != j and col1 in matrix and col2 in matrix[col1]:
                                corr_val = matrix[col1][col2]
                                non_diagonal_corrs.append({
                                    "col1": col1,
                                    "col2": col2,
                                    "value": corr_val
                                })
                
                text = f"✅ Correlation heatmap created!\n  • Saved to: {path}\n  • Shows correlation coefficients between {len(columns)} numeric columns: {', '.join(columns) if columns else 'all numeric columns'}\n\n"
                
                if non_diagonal_corrs:
                    # Sort by absolute value and show top correlations
                    non_diagonal_corrs.sort(key=lambda x: abs(x["value"]), reverse=True)
                    text += "Top Correlations (from the matrix):\n"
                    for corr in non_diagonal_corrs[:10]:  # Show top 10
                        strength = "strong" if abs(corr["value"]) >= 0.7 else "moderate" if abs(corr["value"]) >= 0.5 else "weak"
                        text += f"  • {corr['col1']} vs {corr['col2']}: {corr['value']:.3f} ({strength})\n"
                
                text += "\nNote: All correlation values are in the heatmap. Diagonal values (1.0) are self-correlations."
                return text
            return "Correlation matrix plot generation failed."
        
        elif tool_name == "compare_columns":
            col1 = result.get("column1", "N/A")
            col2 = result.get("column2", "N/A")
            if "correlation" in result:
                corr = result["correlation"]
                text = f"Comparison: {col1} vs {col2}\n"
                text += f"  • Correlation: {corr:.3f}\n"
                return text
            return "Comparison data not available."
        
        elif tool_name == "get_data_sample":
            sample = result.get("sample", [])
            sample_type = result.get("sample_type", "head")
            if sample:
                text = f"Sample data ({sample_type}, {len(sample)} rows):\n"
                # Show first few rows as example
                for i, row in enumerate(sample[:3]):
                    text += f"  Row {i+1}: {row}\n"
                return text
            return "No sample data available."
        
        elif tool_name == "group_by_stats":
            group_by = result.get("group_by", "N/A")
            column = result.get("column", "N/A")
            groups = result.get("groups", {})
            if groups:
                text = f"Statistics for {column} grouped by {group_by}:\n\n"
                # Sort by mean (or max) to show highest values first
                sorted_groups = sorted(groups.items(), key=lambda x: x[1].get('mean', 0), reverse=True)
                for group_name, stats in sorted_groups:
                    mean_val = stats.get('mean', 0)
                    count = stats.get('count', 0)
                    min_val = stats.get('min', 0)
                    max_val = stats.get('max', 0)
                    text += f"  • {group_name}:\n"
                    text += f"    - Count: {count}\n"
                    text += f"    - Average: {mean_val:,.2f}\n"
                    text += f"    - Range: {min_val:,.2f} to {max_val:,.2f}\n\n"
                return text
            return "No grouped statistics available."
        
        elif tool_name == "detect_outliers":
            outliers = result.get("outliers", [])
            method = result.get("method", "iqr")
            if outliers:
                text = f"Outliers detected ({method} method, {len(outliers)} found):\n"
                for outlier in outliers[:5]:  # Show top 5
                    text += f"  • {outlier}\n"
                return text
            return "No outliers detected."
        
        elif tool_name == "data_quality_report":
            issues = result.get("issues", [])
            quality_score = result.get("quality_score", "N/A")
            text = f"Data Quality Report:\n"
            text += f"  • Quality Score: {quality_score}\n"
            if issues:
                text += f"  • Issues found: {len(issues)}\n"
                for issue in issues[:5]:  # Show top 5
                    text += f"    - {issue}\n"
            return text
        
        # Default fallback
        return json.dumps(result, indent=2)
