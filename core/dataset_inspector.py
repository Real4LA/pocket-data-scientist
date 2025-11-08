from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
from typing import Dict, Any, Union
from datetime import datetime

# Project root is one level up from core/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

class DatasetInspector:
    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        self.df = pd.read_csv(self.csv_path)
        # Store project root for plot paths (plots/ directory is at project root)
        self.project_root = _PROJECT_ROOT

    def get_data_overview(self) -> Dict[str, Any]:
        """Comprehensive dataset overview including memory usage."""
        columns = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        
        # Detect date columns by name (since pandas reads them as object unless parsed)
        date_cols = []
        for col in columns.keys():
            col_lower = col.lower()
            if ("date" in col_lower or "time" in col_lower) and "datetime" not in str(columns[col]).lower():
                # Try to detect if it's actually a date by sampling
                try:
                    sample = self.df[col].dropna().head(5)
                    if len(sample) > 0:
                        # Check if values look like dates
                        sample_str = sample.astype(str)
                        if any("/" in str(v) or "-" in str(v) for v in sample_str if len(str(v)) > 5):
                            date_cols.append(col)
                except:
                    pass
        
        # Also check dtype-based detection
        dtype_date_cols = [col for col, dtype in columns.items() if "date" in dtype.lower() or "datetime" in dtype.lower()]
        date_cols = list(set(date_cols + dtype_date_cols))
        
        return {
            "shape": {"rows": int(len(self.df)), "columns": int(len(columns))},
            "columns": list(columns.keys()),
            "dtypes": columns,
            "memory_usage_mb": round(float(self.df.memory_usage(deep=True).sum() / 1024**2), 2),
            "numeric_columns": [col for col, dtype in columns.items() if "int" in dtype or "float" in dtype],
            "categorical_columns": [col for col, dtype in columns.items() if "object" in dtype and col not in date_cols],
            "date_columns": date_cols,
            "duplicate_rows": int(self.df.duplicated().sum()),
        }

    def list_columns(self) -> Dict[str, Any]:
        """Returns column information with additional dataset context."""
        overview = self.get_data_overview()
        return {
            "columns": {col: overview["dtypes"][col] for col in overview["columns"]},
            "total_rows": overview["shape"]["rows"],
            "total_columns": overview["shape"]["columns"],
            "numeric_columns": overview["numeric_columns"],
            "text_columns": overview["categorical_columns"],
            "date_columns": overview["date_columns"],
        }

    def summary_stats(self) -> Dict[str, Any]:
        """Returns comprehensive numeric statistics with insights, skewness, and kurtosis."""
        # Use ALL data - no sampling
        numeric_df = self.df.select_dtypes(include="number")
        if numeric_df.empty:
            return {"stats": {}, "insights": "No numeric columns found in the dataset."}
        
        # Verify we're using all rows
        total_rows = len(self.df)
        
        desc = numeric_df.describe()
        rows = ["count", "mean", "std", "min", "max"]
        stats: Dict[str, Dict[str, float]] = {}
        insights = []
        
        for col in desc.columns:
            col_data = numeric_df[col].dropna()
            # Get stats from describe() which uses ALL data
            stats[col] = {r: float(desc.loc[r, col]) for r in rows if r in desc.index}
            
            # CRITICAL: Double-check max/min using direct calculation to ensure accuracy
            # This ensures we get the exact maximum and minimum values from the actual data
            actual_max = float(col_data.max())
            actual_min = float(col_data.min())
            # Verify we're using all non-null values
            non_null_count = len(col_data)
            
            # Ensure we're using the actual values - override any potential rounding from describe()
            if "max" in stats[col]:
                stats[col]["max"] = actual_max
            if "min" in stats[col]:
                stats[col]["min"] = actual_min
            # Update count to reflect actual non-null values
            if "count" in stats[col]:
                stats[col]["count"] = float(non_null_count)
            
            # Add skewness and kurtosis
            stats[col]["skewness"] = float(col_data.skew())
            stats[col]["kurtosis"] = float(col_data.kurtosis())
            
            # Add additional insights
            mean_val = stats[col]["mean"]
            std_val = stats[col]["std"]
            min_val = stats[col]["min"]
            max_val = stats[col]["max"]
            
            # Calculate coefficient of variation (relative spread)
            cv = (std_val / mean_val * 100) if mean_val != 0 else 0
            
            # Detect potential outliers (values beyond 2 std from mean)
            outliers = col_data[(col_data < mean_val - 2*std_val) | (col_data > mean_val + 2*std_val)]
            outlier_count = len(outliers)
            
            # Distribution insights
            if cv < 15:
                spread_desc = "relatively consistent"
            elif cv < 50:
                spread_desc = "moderately variable"
            else:
                spread_desc = "highly variable"
            
            col_insights = {
                "column": col,
                "coefficient_of_variation": round(cv, 2),
                "spread_description": spread_desc,
                "outlier_count": int(outlier_count),
                "outlier_percentage": round(outlier_count / len(col_data) * 100, 1) if len(col_data) > 0 else 0,
                "range": round(max_val - min_val, 2),
            }
            insights.append(col_insights)
        
        # Use actual row count to ensure accuracy
        total_records = len(self.df)
        
        return {
            "stats": stats,
            "insights": insights,
            "total_numeric_columns": len(stats),
            "total_records": total_records,
            "dataset_rows": total_records,  # Explicit row count for verification
        }

    def get_column_summary(self, column: str) -> Dict[str, Any]:
        """Returns detailed column summary with insights and patterns."""
        if column not in self.df.columns:
            raise ValueError(f"Unknown column: {column}")
        s = self.df[column].dropna()
        total = len(self.df)
        non_null = int(s.notna().sum())
        nulls = int(self.df[column].isna().sum())
        unique = int(s.nunique(dropna=True))
        
        base = {
            "column_name": column,
            "dtype": str(self.df[column].dtype),
            "total_rows": total,
            "non_null": non_null,
            "nulls": nulls,
            "null_percentage": round(nulls / total * 100, 2) if total > 0 else 0,
            "unique": unique,
            "unique_percentage": round(unique / non_null * 100, 2) if non_null > 0 else 0,
        }
        
        if pd.api.types.is_numeric_dtype(s):
            base["numeric"] = {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
                "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            }
            # Add insights
            mean_val = base["numeric"]["mean"]
            median_val = base["numeric"]["median"]
            if abs(mean_val - median_val) / mean_val > 0.1 if mean_val != 0 else False:
                base["insights"] = {
                    "distribution": "skewed" if mean_val > median_val else "left-skewed",
                    "note": "Mean and median differ significantly, suggesting a skewed distribution"
                }
            else:
                base["insights"] = {
                    "distribution": "approximately symmetric",
                    "note": "Mean and median are close, suggesting a symmetric distribution"
                }
        else:
            vc = s.value_counts(dropna=True)
            top_5 = vc.head(5)
            base["top_values"] = [{"value": str(idx), "count": int(cnt), "percentage": round(cnt / non_null * 100, 2)} 
                                 for idx, cnt in top_5.items()]
            # Add insights for categorical data
            if unique < 10:
                base["insights"] = {
                    "type": "categorical",
                    "note": f"Low cardinality ({unique} unique values) - good for grouping/analysis"
                }
            elif unique / non_null > 0.9:
                base["insights"] = {
                    "type": "highly_unique",
                    "note": f"Very high uniqueness ({unique}/{non_null}) - likely an identifier"
                }
            else:
                base["insights"] = {
                    "type": "categorical",
                    "note": f"Moderate cardinality - {unique} unique values out of {non_null} total"
                }
        
        return base


    def correlation_matrix(self, method: str = "pearson", threshold: float = 0.5) -> Dict[str, Any]:
        """Calculate correlation matrix and identify strong correlations."""
        numeric_df = self.df.select_dtypes(include="number")
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis", "matrix": {}}
        
        corr_matrix = numeric_df.corr(method=method)
        
        # Find strong correlations
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_corrs.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": float(corr_val),
                        "strength": "strong" if abs(corr_val) >= 0.7 else "moderate"
                    })
        
        return {
            "method": method,
            "matrix": {col: {other_col: float(corr_matrix.loc[col, other_col]) 
                            for other_col in corr_matrix.columns} 
                      for col in corr_matrix.index},
            "columns": list(corr_matrix.columns),
            "total_pairs": len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2,
            "strong_correlations": sorted(strong_corrs, key=lambda x: abs(x['correlation']), reverse=True),
        }

    def plot_correlation_matrix(self) -> str:
        """Create a heatmap visualization of the correlation matrix."""
        numeric_df = self.df.select_dtypes(include="number")
        if numeric_df.empty or len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation matrix")
        
        import seaborn as sns
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
        plt.title("Correlation Matrix", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plots_dir = self.project_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        suffix = f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        out_path = plots_dir / f"correlation_matrix{suffix}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        # Return relative path from project root for portability
        rel_path = out_path.relative_to(self.project_root)
        return str(rel_path.as_posix())  # Use forward slashes for cross-platform compatibility

    def compare_columns(self, column1: str, column2: str) -> Dict[str, Any]:
        """Compare two columns and return statistics."""
        if column1 not in self.df.columns or column2 not in self.df.columns:
            raise ValueError(f"One or both columns not found: {column1}, {column2}")
        
        col1_data = self.df[column1].dropna()
        col2_data = self.df[column2].dropna()
        
        result = {
            "column1": column1,
            "column2": column2,
            "column1_type": str(self.df[column1].dtype),
            "column2_type": str(self.df[column2].dtype),
            "common_rows": int(len(self.df[[column1, column2]].dropna())),
        }
        
        if pd.api.types.is_numeric_dtype(col1_data) and pd.api.types.is_numeric_dtype(col2_data):
            # Numeric comparison
            pairs = self.df[[column1, column2]].dropna()
            if len(pairs) > 0:
                correlation = float(pairs[column1].corr(pairs[column2]))
                result["correlation"] = correlation
                result["correlation_strength"] = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
        
        return result

    def get_data_sample(self, n: int = 10, sample_type: str = "head") -> Dict[str, Any]:
        """Get sample rows from the dataset (head, tail, or random)."""
        n = min(n, len(self.df))
        
        if sample_type == "head":
            sample_df = self.df.head(n)
        elif sample_type == "tail":
            sample_df = self.df.tail(n)
        elif sample_type == "random":
            sample_df = self.df.sample(n=n, random_state=42)
        else:
            raise ValueError(f"Invalid sample_type: {sample_type}. Use 'head', 'tail', or 'random'")
        
        return {
            "sample_type": sample_type,
            "sample_size": len(sample_df),
            "total_rows": len(self.df),
            "columns": list(sample_df.columns),
            "data": sample_df.to_dict(orient='records'),
        }

    def group_by_stats(self, group_by: str, column: str) -> Dict[str, Any]:
        """Group by a column and calculate statistics for another column."""
        if group_by not in self.df.columns:
            raise ValueError(f"Group column not found: {group_by}")
        if column not in self.df.columns:
            raise ValueError(f"Stats column not found: {column}")
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Stats column must be numeric: {column}")
        
        grouped = self.df.groupby(group_by)[column].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        
        return {
            "group_by": group_by,
            "column": column,
            "groups": {str(idx): {
                "count": int(row['count']),
                "mean": float(row['mean']),
                "std": float(row['std']) if pd.notna(row['std']) else 0.0,
                "min": float(row['min']),
                "max": float(row['max']),
            } for idx, row in grouped.iterrows()},
            "total_groups": len(grouped),
        }

    def detect_outliers(self, column: str, method: str = "iqr") -> Dict[str, Any]:
        """Detect outliers in a numeric column."""
        if column not in self.df.columns:
            raise ValueError(f"Column not found: {column}")
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column must be numeric: {column}")
        
        s = self.df[column].dropna()
        
        if method == "iqr":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = s[(s < lower_bound) | (s > upper_bound)]
        else:  # z-score method
            mean_val = s.mean()
            std_val = s.std()
            outliers = s[abs(s - mean_val) > 2 * std_val]
            lower_bound = mean_val - 2 * std_val
            upper_bound = mean_val + 2 * std_val
        
        return {
            "column": column,
            "method": method,
            "outlier_count": int(len(outliers)),
            "outlier_percentage": round(len(outliers) / len(s) * 100, 2),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_values": [float(v) for v in outliers.head(10).tolist()],
        }

    def data_quality_report(self) -> Dict[str, Any]:
        """Generate a comprehensive data quality report."""
        report = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "columns": {},
        }
        
        for col in self.df.columns:
            s = self.df[col]
            col_report = {
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "nulls": int(s.isna().sum()),
                "null_percentage": round(s.isna().sum() / len(s) * 100, 2),
                "unique": int(s.nunique(dropna=True)),
            }
            
            if pd.api.types.is_numeric_dtype(s):
                col_report["numeric_stats"] = {
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std()),
                }
            
            report["columns"][col] = col_report
        
        return report

    def plot_column(self, column: str, kind: str = "auto", y: str | None = None, group_by: str | None = None, bins: int = 30) -> str:
        """Enhanced plot function with more plot types."""
        if column not in self.df.columns:
            raise ValueError(f"Unknown column: {column}")

        s = self.df[column].dropna()
        plt.figure(figsize=(10, 6))

        # Auto-detect type if not provided
        if kind == "auto":
            kind = "hist" if pd.api.types.is_numeric_dtype(s) else "bar"

        plots_dir = self.project_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        suffix = f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        # For scatter plots, include y column in filename
        if kind == "scatter" and y:
            out_path = plots_dir / f"{column}_vs_{y}_{kind}{suffix}.png"
        else:
            out_path = plots_dir / f"{column}_{kind}{suffix}.png"

        if kind == "hist":
            if not pd.api.types.is_numeric_dtype(s):
                raise ValueError(f"{column} must be numeric for histogram")
            s.plot(kind="hist", bins=bins, edgecolor="black", alpha=0.7)
            plt.title(f"Histogram: {column}", fontsize=14, fontweight='bold')
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(axis='y', alpha=0.3)

        elif kind == "bar":
            top_values = s.value_counts().head(15)
            top_values.plot(kind="bar", color='steelblue', edgecolor='black')
            plt.title(f"Top Categories: {column}", fontsize=14, fontweight='bold')
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)

        elif kind == "box":
            if not pd.api.types.is_numeric_dtype(s):
                raise ValueError(f"{column} must be numeric for boxplot")
            
            if group_by:
                if group_by not in self.df.columns:
                    raise ValueError(f"Group column '{group_by}' not found")
                self.df.boxplot(column=column, by=group_by, grid=True, ax=plt.gca())
                plt.title(f"Box Plot: {column} by {group_by}", fontsize=14, fontweight='bold')
                plt.suptitle('')  # Remove automatic title
                plt.xlabel(group_by, fontsize=12)
            else:
                self.df[[column]].boxplot(grid=True)
                plt.title(f"Boxplot: {column}", fontsize=14, fontweight='bold')
                plt.ylabel(column, fontsize=12)
            plt.grid(axis='y', alpha=0.3)

        elif kind == "scatter":
            if y is None:
                raise ValueError("Scatterplot requires 'y' column name")
            if y not in self.df.columns:
                raise ValueError(f"Unknown y column: {y}")
            pairs = self.df[[column, y]].dropna()
            if len(pairs) < 5 or pairs[column].std() == 0 or pairs[y].std() == 0:
                raise ValueError("Scatter not informative: need ≥5 pairs and non-constant columns.")
            
            plt.scatter(pairs[column], pairs[y], alpha=0.6, s=50)
            
            # Add trendline if requested (default True for scatter)
            try:
                import numpy as np
                z = np.polyfit(pairs[column], pairs[y], 1)
                p = np.poly1d(z)
                plt.plot(pairs[column], p(pairs[column]), "r--", alpha=0.8, linewidth=2, label='Trend line')
                plt.legend()
            except:
                pass
            
            plt.title(f"Scatter Plot: {column} vs {y}", fontsize=14, fontweight='bold')
            plt.xlabel(column, fontsize=12)
            plt.ylabel(y, fontsize=12)
            plt.grid(alpha=0.3)

        elif kind == "line":
            if not pd.api.types.is_numeric_dtype(s):
                raise ValueError(f"{column} must be numeric for line plot")
            s.plot(kind="line", marker='o', markersize=4)
            plt.title(f"Line Plot: {column}", fontsize=14, fontweight='bold')
            plt.xlabel("Index", fontsize=12)
            plt.ylabel(column, fontsize=12)
            plt.grid(alpha=0.3)

        elif kind == "density":
            if not pd.api.types.is_numeric_dtype(s):
                raise ValueError(f"{column} must be numeric for density plot")
            s.plot(kind="density", linewidth=2)
            plt.title(f"Density Plot: {column}", fontsize=14, fontweight='bold')
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.grid(alpha=0.3)

        else:
            raise ValueError(f"Unknown plot kind: {kind}. Available: hist, bar, box, scatter, line, density")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        # Return relative path from project root for portability
        rel_path = out_path.relative_to(self.project_root)
        return str(rel_path.as_posix())  # Use forward slashes for cross-platform compatibility