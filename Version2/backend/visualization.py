import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional

class VisualizationManager:
    @staticmethod
    def _is_identifier_like_column(col_name: str) -> bool:
        """
        Checks if a column name suggests it's primarily an identifier.
        """
        name_lower = str(col_name).lower()

        # Exact matches for common ID names
        if name_lower in ["id", "uuid", "guid", "identifier", "key", "rowid", 
                          "pk", "primary_key", "foreign_key", "fk", "serialnumber", "serial_number"]:
            return True

        # Prefixes that strongly indicate an identifier
        if name_lower.startswith("id_") or \
           name_lower.startswith("key_") or \
           name_lower.startswith("identifier_") or \
           name_lower.startswith("pk_") or \
           name_lower.startswith("fk_") or \
           name_lower.startswith("serial_"):
            return True

        # Suffixes that strongly indicate an identifier (e.g., customer_id, order_key)
        if name_lower.endswith("_id") or \
           name_lower.endswith("_key") or \
           name_lower.endswith("_identifier") or \
           name_lower.endswith("_serial"):
            return True

        # For concatenated IDs like 'productid', 'orderkey' but not common English words
        # Common English words that end with "id" or "key" that are NOT identifiers.
        non_id_words_ending_with_id = [
            'acid', 'afraid', 'aid', 'amid', 'apartheid', 'aside', 'avoid', 'award',
            'bid', 'carotid', 'david', 'did', 'fluid', 'forbid', 'grid', 'hybrid',
            'invalid', 'kid', 'madrid', 'maid', 'mid', 'overpaid', 'paid', 'period',
            'pyramid', 'raid', 'rapid', 'rid', 'rigid', 'solid', 'squid', 'stupid',
            'subsid', 'thyroid', 'torpid', 'tripod', 'unavoid', 'unpaid', 'valid', 'void', 'worldwid'
        ]
        # non_id_words_ending_with_key = ['donkey', 'hockey', 'jockey', 'lackey', 'monkey', 'turkey', 'whiskey'] # Less of an issue

        if name_lower.endswith("id") and len(name_lower) > 2: # Exclude "id" itself (already covered)
            if name_lower not in non_id_words_ending_with_id:
                return True
        
        if name_lower.endswith("key") and len(name_lower) > 3: # Exclude "key" itself (already covered)
            # if name_lower not in non_id_words_ending_with_key: # Typically not needed for "key"
                return True

        # Contains "identifier" or "serial" as part of the name
        if "identifier" in name_lower or "serial" in name_lower:
            return True
            
        return False

    @staticmethod
    def auto_visualize(df: pd.DataFrame, question: str) -> Optional[go.Figure]:
        """Automatically create appropriate visualization based on data"""
        if df.empty or len(df) == 1:
            return None
        
        all_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not all_numeric_cols: # No numeric columns at all
            return None

        # Filter out columns that are primarily identifiers
        plottable_numeric_cols = [
            col for col in all_numeric_cols 
            if not VisualizationManager._is_identifier_like_column(col)
        ]

        # If all numeric columns were identifier-like (or no numeric columns left after filtering)
        if not plottable_numeric_cols:
            return None
            
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        fig: Optional[go.Figure] = None

        # Time series plot
        if datetime_cols and plottable_numeric_cols:
            x_time_col = datetime_cols[0]
            y_val_col = plottable_numeric_cols[0]
            # Ensure data is sorted by time for line charts
            df_sorted_time = df.sort_values(by=x_time_col)
            fig = px.line(df_sorted_time, x=x_time_col, y=y_val_col,
                          title=f"Time Series: {y_val_col} over {x_time_col}")
        
        # Bar chart for categorical vs numeric
        elif categorical_cols and plottable_numeric_cols and len(df) <= 50:
            x_cat_col = categorical_cols[0]
            y_val_col = plottable_numeric_cols[0]
            # Sort DataFrame by the value column (y_val_col) in descending order
            df_sorted_bar = df.sort_values(by=y_val_col, ascending=False)
            fig = px.bar(df_sorted_bar, x=x_cat_col, y=y_val_col,
                         title=f"{y_val_col} by {x_cat_col} (Sorted by {y_val_col})")
        
        # Scatter plot for two numeric columns
        elif len(plottable_numeric_cols) >= 2:
            fig = px.scatter(df, x=plottable_numeric_cols[0], y=plottable_numeric_cols[1],
                            title=f"{plottable_numeric_cols[1]} vs {plottable_numeric_cols[0]}")
        
        # Histogram for single numeric column
        elif len(plottable_numeric_cols) == 1:
            fig = px.histogram(df, x=plottable_numeric_cols[0],
                              title=f"Distribution of {plottable_numeric_cols[0]}")
        
        # Add general layout updates for better readability if a figure is created
        if fig:
            fig.update_layout(
                title_x=0.5, # Center title
                legend_title_text='', # Remove legend title if any
            )

        return fig
