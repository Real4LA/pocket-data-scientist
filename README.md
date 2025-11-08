# 🧪 Pocket Data Scientist

A powerful AI-powered data analysis tool that combines Streamlit, Ollama (Llama3), and Model Context Protocol (MCP) to provide an interactive, conversational interface for exploring and analyzing CSV datasets.

## 🎥 Demo

Watch the demo video on YouTube to see Pocket Data Scientist in action: [YouTube Demo](https://www.youtube.com/watch?v=D4lXg_nML5Y)

## ✨ Features

### 🤖 AI-Powered Analysis

- **Conversational Interface**: Ask questions in natural language about your dataset
- **Intelligent Tool Selection**: AI automatically selects the right analysis tools based on your questions
- **Few-Shot Learning**: Optimized prompts with examples for better tool utilization
- **Exact Value Reporting**: Ensures precise data reporting without rounding or estimation

### 📊 Data Analysis Capabilities

- **Dataset Overview**: Get comprehensive information about your dataset structure
- **Summary Statistics**: Detailed statistics for all numeric columns
- **Column Analysis**: Explore individual columns with top values and distributions
- **Grouped Statistics**: Analyze data grouped by categories
- **Correlation Analysis**: Discover relationships between numeric variables
- **Data Quality Reports**: Identify missing values, duplicates, and data quality issues

### 📈 Visualization

- **Scatter Plots**: Visualize relationships between two variables
- **Bar Charts**: Display category distributions
- **Histograms**: Show value distributions
- **Correlation Heatmaps**: Interactive correlation matrix visualizations
- **CSV Data Viewer**: Browse your entire dataset in an interactive table

### 🎨 User Interface

- **Modern Streamlit UI**: Clean, intuitive interface with sidebar controls
- **Tabbed Interface**: Separate tabs for Chat and Data Viewer
- **Real-time Updates**: See results as they're generated
- **Tool Usage Tracking**: Expandable section showing all tool calls and results

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Llama3 model downloaded (default: `llama3:8b`)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Real4LA/pocket-data-scientist
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install and start Ollama**

   - Download from [ollama.ai](https://ollama.ai/)
   - Pull the Llama3 model:
     ```bash
     ollama pull llama3:8b
     ```

4. **Start the application**

   ```bash
   streamlit run app/chat_app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

## 📖 Usage Guide

### Getting Started

1. **Upload a CSV file**

   - Click "Upload CSV File" in the sidebar
   - Select your dataset file
   - Wait for the "Loaded" confirmation

2. **Explore your data**

   - Use the **Data Viewer** tab to browse your dataset
   - Switch to the **Chat** tab to ask questions

3. **Ask questions**
   - Type your question in the input field at the bottom
   - Examples:
     - "What is this dataset about?"
     - "Show me summary statistics"
     - "What is the highest paying job?"
     - "Visualize the relationship between age and salary"
     - "Plot correlation matrix"

### Example Queries

#### Basic Information

```
What is this dataset about?
Show me all columns
What are the summary statistics?
```

#### Analysis Questions

```
What is the highest paying job?
What is the average salary by job title?
What are the most common cities?
What is the relationship between age and salary?
```

#### Visualizations

```
Plot correlation matrix
Visualize age vs salary
Create a histogram of salary distribution
Show a bar chart of job title counts
```

#### Advanced Analysis

```
What are the top 2 correlated features?
What is the most contributing factor in salary elevation?
Detect outliers in salary
```

## 🛠️ Available Tools

The system provides the following analysis tools:

### Data Exploration

- **`list_columns()`**: Get column names, types, and dataset structure
- **`get_data_overview()`**: Comprehensive dataset overview (shape, memory, column types)
- **`summary_stats()`**: Statistics for all numeric columns (mean, std, min, max, etc.)
- **`get_data_sample()`**: Get sample rows from the dataset

### Column Analysis

- **`get_column_summary(column)`**: Detailed analysis of a specific column
  - Top values and counts
  - Numeric statistics (if applicable)
  - Null percentage
  - Unique value percentage

### Statistical Analysis

- **`group_by_stats(group_by, column)`**: Statistics grouped by a category column
  - Mean, std, min, max per group
  - Count per group
- **`correlation_matrix()`**: Correlation coefficients between numeric columns
- **`compare_columns(column1, column2)`**: Detailed comparison between two columns

### Visualization

- **`plot_column(column, kind, y)`**: Create various plot types
  - `kind='scatter'` with `y`: Scatter plot (X vs Y)
  - `kind='bar'`: Bar chart for category counts
  - `kind='hist'`: Histogram for distributions
  - `kind='auto'`: Automatic plot type selection
- **`plot_correlation_matrix()`**: Heatmap visualization of correlation matrix

### Data Quality

- **`detect_outliers(column, method)`**: Identify outliers using IQR or Z-score
- **`data_quality_report()`**: Comprehensive data quality assessment

## 🏗️ Architecture

### Components

```
dataset-inspector-mcp/
├── app/
│   └── chat_app.py          # Streamlit UI application
├── agents/
│   └── agent.py             # AI agent with tool selection logic
├── core/
│   ├── dataset_inspector.py # Core data analysis engine
│   ├── server_mcp.py        # MCP server exposing tools
│   └── tool_client.py       # MCP client for tool communication
├── data/
│   └── data.csv             # Sample dataset
├── plots/                   # Generated visualizations
└── uploads/                 # User-uploaded datasets
```

### How It Works

1. **User Input**: User types a question in the Streamlit interface

2. **Agent Processing**: The `ToolAwareAgent` analyzes the question and decides which tools to call

3. **Tool Execution**: Tools are executed via MCP (Model Context Protocol) through the `ToolClient`

4. **Data Analysis**: The `DatasetInspector` performs the actual data analysis using pandas

5. **Response Generation**: The agent formats the tool results into a natural language response

6. **Display**: Results are shown in the UI, including plots, tables, and text responses

### MCP Integration

The project uses FastMCP to expose data analysis tools via the Model Context Protocol:

- Tools are defined in `core/server_mcp.py`
- The `ToolClient` communicates with the MCP server via stdio
- This allows the AI agent to dynamically call tools based on user questions

## ⚙️ Configuration

### Model Configuration

Edit `app/chat_app.py` to change the default model:

```python
DEFAULT_MODEL = "llama3:8b"  # Change to your preferred model
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
```

### Supported Models

Any Ollama model that supports chat completions. Recommended:

- `llama3:8b` (default)
- `llama3.2:3b` (faster, less accurate)
- `llama3:70b` (slower, more accurate)

## 📝 Project Structure

```
dataset-inspector-mcp/
├── agents/
│   ├── __init__.py
│   └── agent.py              # AI agent with prompt engineering
├── app/
│   ├── __init__.py
│   └── chat_app.py           # Streamlit web application
├── core/
│   ├── __init__.py
│   ├── dataset_inspector.py  # Data analysis engine
│   ├── server_mcp.py         # MCP server
│   └── tool_client.py        # MCP client
├── data/
│   └── data.csv              # Sample dataset
├── plots/                    # Generated visualizations
├── uploads/                  # User uploads directory
├── local.json                # MCP configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔒 Data Privacy

- All data processing happens locally on your machine
- No data is sent to external servers (except Ollama API calls)
- Uploaded files are stored in the `uploads/` directory
- Generated plots are saved in the `plots/` directory

## 📚 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting library
- **seaborn**: Statistical visualizations
- **fastmcp**: Model Context Protocol server
- **requests**: HTTP client for Ollama API

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Ollama](https://ollama.ai/) and Llama3
- Uses [FastMCP](https://github.com/jlowin/fastmcp) for MCP integration

## 📞 Support

For issues, questions, or contributions, please open an issue on the repository.

---

**Happy Data Analyzing! 🎉**
