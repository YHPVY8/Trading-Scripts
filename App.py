import dash
from dash import Dash, dash_table, html
import pandas as pd
from supabase import create_client

SUPABASE_URL = "YOUR_URL"
SUPABASE_KEY = "YOUR_KEY"
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# Fetch last 1000 rows of daily_es
res = sb.table("daily_es").select("*").order("time", desc=True).limit(1000).execute()
df = pd.DataFrame(res.data).sort_values("time")
df = df.round(2)

app = Dash(__name__)
app.layout = html.Div([
    html.H1("Trading Dashboard"),
    dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": i, "id": i, "type": "numeric" if pd.api.types.is_numeric_dtype(df[i]) else "text"} for i in df.columns],
        page_size=25,
        style_table={"overflowX": "auto"},
        style_cell={"minWidth": "80px", "whiteSpace": "normal"},
        style_data_conditional=[
            {"if": {"filter_query": "{Hit Pivot} = True", "column_id": "Hit Pivot"}, "backgroundColor": "#98FB98"}
        ]
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
