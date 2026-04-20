import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import textwrap
 
 
# Colour palette
# These are the colours used for the dashboard and surrounding asthetics apart from the heatmap
# The heatmap uses Cividis colours, to be accessible to people suffering from Deuteranopia and Protanopia

NHS_BLUE       = "#005EB8"
NHS_DARK_BLUE  = "#003087"
NHS_LIGHT_BLUE = "#41B6E6"
WHITE          = "#FFFFFF"
LIGHT_GREY     = "#E8EDEE"
DARK_GREY      = "#425563"
YELLOW         = "#FFB81C"
RED            = "#8B0000"
 
 
# Data loading
df = pd.read_csv("final_nhs_full.csv")
 
# Sorted list of unique diagnostic categories used to populate the dropdown
categories = sorted(df["category"].dropna().unique())
 
 
# Dashboard initialisation
app = Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap"
])
 
 
# Functions - used throughout the dashboard

def wrap_label(text, width=20):
    """
    Breaks down long pieces of text into sections,
    so the text can appear on multiple lines.
    """
    return "<br>".join(textwrap.wrap(text, width))
 
def card_style(width, extra=None):
    """
    Builds a base CSS style dictionary with standard card appearance
    (white background, rounded corners, drop shadow, padding).
    Additional styles can be passed via the extra parameter and are
    merged on top of the base using dict.update().
    """
    extra = extra or {}
    base = {
        "display"         : "inline-block",
        "verticalAlign"   : "top",
        "backgroundColor" : WHITE,
        "borderRadius"    : "6px",
        "boxShadow"       : "0 2px 8px rgba(0,0,0,0.1)",
        "padding"         : "16px",
        "boxSizing"       : "border-box",
        "width"           : width
    }
    base.update(extra)
    return base
 
 
def analytics(filtered, sort_method):
    """
    Returns a list of up to 5 diagnosis codes from the filtered DataFrame
    based on the selected sort method. Supports ranking by:
      - 'admissions'       : highest total admission volume
      - 'bottom_admissions': lowest admission volume (min 100 admissions)
      - 'highest_los'      : highest mean length of stay
      - 'los_change'       : greatest absolute change in mean LOS over time
    """
 
    if sort_method == "admissions":
        return (filtered.groupby("diagnosis_code")["admissions"]
                .sum().nlargest(5).index.tolist())
 
    elif sort_method == "highest_los":
        return (filtered.groupby("diagnosis_code")["mean_length_stay"]
                .mean().nlargest(5).index.tolist())
 
    elif sort_method == "bottom_admissions":
        totals = filtered.groupby("diagnosis_code")["admissions"].sum()
        return totals[totals > 100].nsmallest(5).index.tolist()
 
    elif sort_method == "los_change":
        def los_change(x):
            """
            Calculates the absolute change in mean length of stay (LOS) for a
            single diagnosis between its earliest and latest recorded year.
            Returns 0 if fewer than 2 years of valid data are available.
            Used by analytics() to rank diagnoses by greatest LOS change over time.
            """
            valid = x.dropna(subset=["mean_length_stay"])
            if len(valid) < 2:
                return 0
            return abs(
                valid.loc[valid["year"].idxmax(), "mean_length_stay"] -
                valid.loc[valid["year"].idxmin(), "mean_length_stay"]
            )
        return (filtered.groupby("diagnosis_code")
                .apply(los_change)
                .nlargest(5).index.tolist())
 
    return []
 
 
def get_ordered_codes(filtered, top5_codes, sort_method):
    """
    Returns the list of diagnosis codes sorted for heatmap row ordering,
    consistent with the active sort method. Extracted from update_heatmap
    to avoid duplicating the ordering logic that already exists in analytics().
    """
    if sort_method == "admissions":
        return (filtered[filtered["diagnosis_code"].isin(top5_codes)]
                .groupby("diagnosis_code")["admissions"]
                .sum()
                .sort_values(ascending=False)
                .index.tolist())
 
    elif sort_method == "bottom_admissions":
        return (filtered[filtered["diagnosis_code"].isin(top5_codes)]
                .groupby("diagnosis_code")["admissions"]
                .sum()
                .sort_values(ascending=True)
                .index.tolist())
 
    elif sort_method == "highest_los":
        return (filtered[filtered["diagnosis_code"].isin(top5_codes)]
                .groupby("diagnosis_code")["mean_length_stay"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist())
 
    elif sort_method == "los_change":
        def los_change_order(x):
            """
            Calculates absolute LOS change between first and last recorded year.
            Used solely for ordering heatmap rows when sort_method is 'los_change'.
            """
            valid = x.dropna(subset=["mean_length_stay"])
            if len(valid) < 2:
                return 0
            return abs(
                valid.loc[valid["year"].idxmax(), "mean_length_stay"] -
                valid.loc[valid["year"].idxmin(), "mean_length_stay"]
            )
        return (filtered[filtered["diagnosis_code"].isin(top5_codes)]
                .groupby("diagnosis_code")
                .apply(los_change_order)
                .sort_values(ascending=False)
                .index.tolist())
 
    return top5_codes
 
 
# Dashboard layout
 
app.layout = html.Div([
 
    # Header bar
    html.Div([
        html.Div([
            html.Span("COMP4037", style={
                "backgroundColor" : WHITE,
                "color"           : NHS_BLUE,
                "fontWeight"      : "800",
                "fontSize"        : "26px",
                "padding"         : "4px 10px",
                "marginRight"     : "16px",
                "letterSpacing"   : "3px",
                "borderRadius"    : "2px"
            }),
            html.Span("NHS Hospital Admission Analysis", style={
                "color"      : WHITE,
                "fontSize"   : "18px",
                "fontWeight" : "400",
            })
        ], style={
            "maxWidth"   : "1400px",
            "margin"     : "0 auto",
            "padding"    : "14px 24px",
            "display"    : "flex",
            "alignItems" : "center"
        })
    ], style={"backgroundColor": NHS_BLUE, "width": "100%"}),
 
    html.Div(style={
        "backgroundColor" : NHS_DARK_BLUE,
        "height"          : "4px",
        "width"           : "100%"
    }),
 
    # Main content container
    html.Div([
 
        # Dashboard description
        html.P(
            "This visualisation dashboard makes use of datasets made available through "
            "NHS Digital, Hospital Episode Statistics for England. "
            "Admitted Patient Care statistics, 1998–2023. "
            "Select a diagnostic category and ranking method to explore admission trends "
            "in conditions.",
            style={
                "color"          : DARK_GREY,
                "fontSize"       : "14px",
                "lineHeight"     : "1.7",
                "borderLeft"     : f"4px solid {NHS_BLUE}",
                "paddingLeft"    : "14px",
                "marginBottom"   : "24px",
                "marginTop"      : "20px",
                "backgroundColor": LIGHT_GREY
            }
        ),
 
        # Category dropdown and sort method buttons
        html.Div([
 
            html.Div([
                html.Label("Select Diagnostic Category:", style={
                    "fontWeight"   : "600",
                    "color"        : NHS_DARK_BLUE,
                    "fontSize"     : "14px",
                    "marginBottom" : "6px",
                    "display"      : "block"
                }),
                dcc.Dropdown(
                    id="category-dropdown",
                    options=[{"label": cat, "value": cat} for cat in categories],
                    value=categories[0],
                    clearable=False,
                    style={
                        "fontSize"    : "14px",
                        "borderRadius": "4px",
                        "border"      : f"2px solid {NHS_BLUE}",
                        "fontFamily"  : "Inter, Arial, sans-serif"
                    }
                )
            ], style={"width": "48%", "display": "inline-block",
                      "verticalAlign": "top", "marginRight": "4%"}),
 
            html.Div([
                html.Label("Rank Conditions By:", style={
                    "fontWeight"   : "600",
                    "color"        : NHS_DARK_BLUE,
                    "fontSize"     : "14px",
                    "marginBottom" : "10px",
                    "display"      : "block"
                }),
                dcc.RadioItems(
                    id="sort-method",
                    options=[
                        {"label": " Top 5 by Admission Volume",    "value": "admissions"},
                        {"label": " Bottom 5 by Admission Volume", "value": "bottom_admissions"},
                        {"label": " Top 5 by Highest Mean LOS",    "value": "highest_los"},
                        {"label": " Top 5 by Greatest LOS Change", "value": "los_change"}
                    ],
                    value="admissions",
                    inline=True,
                    style={
                        "fontSize" : "13px",
                        "color"    : DARK_GREY,
                        "gap"      : "20px"
                    }
                )
            ], style={"width": "48%", "display": "inline-block",
                      "verticalAlign": "top"})
 
        ], style={"marginBottom": "24px"}),
 
        # Summary statistic cards
        html.Div(id="summary-stats", style={"marginBottom": "20px"}),
 
        # Heatmap and line chart (left and right)
        html.Div([
 
            html.Div([
                dcc.Loading(type="dot", color=NHS_BLUE, children=[
                    dcc.Graph(id="heatmap",
                              config={"displayModeBar": False},
                              style={"height": "500px"})
                ])
            ], style=card_style("65%", {"marginRight": "2%", "minHeight": "540px"})),
 
            html.Div([
                dcc.Loading(type="dot", color=NHS_BLUE, children=[
                    dcc.Graph(id="line-chart",
                              config={"displayModeBar": False},
                              style={"height": "500px"})
                ])
            ], style=card_style("33%"))
 
        ], style={"marginBottom": "20px"}),
 
        html.Div([
            html.Div([
                dcc.Loading(type="dot", color=NHS_BLUE, children=[
                    dcc.Graph(id="scatter-plot",
                              config={"displayModeBar": False},
                              style={"height": "550px"})
                ])
            ], style=card_style("100%"))
        ], style={"marginBottom": "20px"}),
 
        # Footer data 
        html.Div([
            html.P(
                "Data source: Office for National Statistics (ONS) — "
                "NHS Hospital Episode Statistics 1998–2023. "
                "Values are suppressed where admission counts fall below "
                "the NHS disclosure threshold. "
                "Conditions ranked by selected method across all years.",
                style={
                    "color"      : DARK_GREY,
                    "fontSize"   : "11px",
                    "marginTop"  : "24px",
                    "borderTop"  : "1px solid #c8d3d8",
                    "paddingTop" : "14px",
                    "lineHeight" : "1.6"
                }
            )
        ])
 
    ], style={
        "maxWidth"        : "1400px",
        "margin"          : "0 auto",
        "padding"         : "0 24px 48px 24px",
        "fontFamily"      : "Inter, Arial, sans-serif",
        "backgroundColor" : LIGHT_GREY,
        "minHeight"       : "100vh"
    }),
 
], style={"backgroundColor": LIGHT_GREY, "fontFamily": "Inter, Arial, sans-serif"})
 
 
# Dash Callbacks
# Updates the dashboard to reflect the users selected criteria.
 
@app.callback(
    Output("summary-stats", "children"),
    Input("category-dropdown", "value"),
    Input("sort-method", "value")
)
def update_summary(selected_category, sort_method):
    """
    Dash callback that fires whenever the category dropdown or sort method changes.
    Computes four summary statistics for the selected conditions and renders them
    as a row of styled cards: total admissions, average LOS, most improved
    condition by LOS reduction, and the condition with the longest stay in the
    most recent year.
    """
 
    filtered = df[df["category"] == selected_category]
    top5     = analytics(filtered, sort_method)
    top5_df  = filtered[filtered["diagnosis_code"].isin(top5)]
 
    # Aggregate totals across all years for selected conditions
    total_admissions = top5_df["admissions"].sum()
    avg_los          = top5_df["mean_length_stay"].mean()
 
    def pct_change(x):
        """
        Calculates the percentage change in mean LOS between the first and last
        recorded year for a single diagnosis. Guards against missing data and
        division by zero.
        """
        valid = x.dropna(subset=["mean_length_stay"]).sort_values("year")
        if len(valid) < 2:
            return 0
        first = valid.iloc[0]["mean_length_stay"]
        last  = valid.iloc[-1]["mean_length_stay"]
        if first == 0:
            return 0
        return ((last - first) / first) * 100
 
    # Compute per-diagnosis LOS percentage change and identify the most improved
    changes = (top5_df.groupby("diagnosis_code")
               .apply(pct_change)
               .reset_index())
    changes.columns = ["diagnosis_code", "pct_change"]
 
    most_improved_code = changes.loc[changes["pct_change"].idxmin(), "diagnosis_code"]
    most_improved_pct  = changes.loc[changes["pct_change"].idxmin(), "pct_change"]
    most_improved_desc = (filtered[filtered["diagnosis_code"] == most_improved_code]
                          ["description"].iloc[0])
    if len(most_improved_desc) > 30:
        most_improved_desc = most_improved_desc[:30] + "..."
 
    # Identify the condition with the longest mean LOS in the most recent year
    latest_year  = top5_df["year"].max()
    latest_df    = top5_df[top5_df["year"] == latest_year]
    longest_idx  = latest_df["mean_length_stay"].idxmax()
    longest_los  = latest_df.loc[longest_idx, "mean_length_stay"]
    longest_desc = latest_df.loc[longest_idx, "description"]
    if len(longest_desc) > 30:
        longest_desc = longest_desc[:30] + "..."
 

   # Statistic cards
   # found above the heatmap, presenting insight into the selected diagnosis category

    def stat_card(label, value, sub=None):
        """
        Returns a styled html.Div representing a single summary statistic card
        with a label, a large primary value, and an optional subtitle.
        """
        return html.Div([
            html.P(label, style={
                "fontSize"     : "11px",
                "color"        : DARK_GREY,
                "marginBottom" : "4px",
                "fontWeight"   : "600",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px"
            }),
            html.P(value, style={
                "fontSize"     : "22px",
                "color"        : NHS_DARK_BLUE,
                "fontWeight"   : "700",
                "marginBottom" : "2px",
                "lineHeight"   : "1.2"
            }),
            html.P(sub or "", style={
                "fontSize"  : "11px",
                "color"     : DARK_GREY,
                "marginTop" : "0"
            })
        ], style={
            "display"         : "inline-block",
            "width"           : "23%",
            "marginRight"     : "2%",
            "backgroundColor" : WHITE,
            "borderRadius"    : "6px",
            "boxShadow"       : "0 2px 8px rgba(0,0,0,0.1)",
            "padding"         : "14px 16px",
            "boxSizing"       : "border-box",
            "verticalAlign"   : "top",
            "borderTop"       : f"3px solid {NHS_BLUE}"
        })
 
    return html.Div([
        stat_card(
            "Total Admissions (Top 5)",
            f"{total_admissions:,.0f}",
            "Across all years 1998–2023"
        ),
        stat_card(
            "Average Length of Stay",
            f"{avg_los:.1f} days",
            "Mean across top 5 conditions"
        ),
        stat_card(
            "Most Improved",
            most_improved_desc,
            f"{most_improved_pct:.1f}% change since 1998"
        ),
        stat_card(
            f"Longest Stay in {int(latest_year)}",
            longest_desc,
            f"{longest_los:.1f} days average"
        ),
    ])
 

# Heat map 
# Uses Cividis colour scale to ensure accessiblity to the most common colour blind types: Deuteranopia and Protanopia

@app.callback(
    Output("heatmap", "figure"),
    Input("category-dropdown", "value"),
    Input("sort-method", "value")
)
def update_heatmap(selected_category, sort_method):

    filtered   = df[df["category"] == selected_category]
    
    top5_codes = analytics(filtered, sort_method)
    top5_df    = filtered[filtered["diagnosis_code"].isin(top5_codes)]

    code_to_label = (top5_df.groupby("diagnosis_code")["description"]
                    .first()
                    .apply(wrap_label)
                    .to_dict())

    summary = (top5_df.groupby(["year", "diagnosis_code"])["mean_length_stay"]
               .mean().reset_index())

    pivot = summary.pivot(
        index="diagnosis_code", columns="year", values="mean_length_stay")

    desc_matrix = (top5_df.groupby(["year", "diagnosis_code"])["description"]
                   .first().reset_index()
                   .pivot(index="diagnosis_code", columns="year",
                          values="description"))

    if sort_method == "admissions":
        ordered = (filtered[filtered["diagnosis_code"].isin(top5_codes)]
                   .groupby("diagnosis_code")["admissions"]
                   .sum()
                   .sort_values(ascending=False)
                   .index.tolist())

    elif sort_method == "bottom_admissions":
        ordered = (filtered[filtered["diagnosis_code"].isin(top5_codes)]
                   .groupby("diagnosis_code")["admissions"]
                   .sum()
                   .sort_values(ascending=True)
                   .index.tolist())

    elif sort_method == "highest_los":
        ordered = (filtered[filtered["diagnosis_code"].isin(top5_codes)]
                   .groupby("diagnosis_code")["mean_length_stay"]
                   .mean()
                   .sort_values(ascending=False)
                   .index.tolist())

    elif sort_method == "los_change":
        def los_change_order(x):
            valid = x.dropna(subset=["mean_length_stay"])
            if len(valid) < 2:
                return 0
            return abs(
                valid.loc[valid["year"].idxmax(), "mean_length_stay"] -
                valid.loc[valid["year"].idxmin(), "mean_length_stay"]
            )
        ordered = (filtered[filtered["diagnosis_code"].isin(top5_codes)]
                   .groupby("diagnosis_code")
                   .apply(los_change_order)
                   .sort_values(ascending=False)
                   .index.tolist())

    else:
        ordered = top5_codes

    pivot       = pivot.reindex(ordered)
    desc_matrix = desc_matrix.reindex(index=ordered, columns=pivot.columns)

    pivot.index       = pivot.index.map(code_to_label)
    desc_matrix.index = desc_matrix.index.map(code_to_label)

    sort_labels = {
        "admissions"       : "Top 5 by Admission Volume",
        "bottom_admissions": "Bottom 5 by Admission Volume",
        "highest_los"      : "Top 5 by Highest Mean LOS",
        "los_change"       : "Top 5 by Greatest LOS Change"
    }

    sort_labels = sort_labels.get(sort_method)

    fig = px.imshow(
        pivot,
        range_color=[float(pivot.min().min()), float(pivot.max().max())],
        labels=dict(x="Year", y="Diagnosis",
                    color="Mean Length of Stay (Days)"),
        title=f"Mean Length of Stay: {sort_labels} for {selected_category} diagnoses (1998 - 2023)",
        color_continuous_scale="Cividis",
        aspect="auto",
        text_auto=".1f"
    )

    fig.update_layout(
        plot_bgcolor  = WHITE,
        paper_bgcolor = WHITE,
        font=dict(family="Inter, Arial, sans-serif",
                  color=DARK_GREY, size=12),
        title=dict(font=dict(size=15, color=NHS_DARK_BLUE), x=0.01),
        coloraxis_colorbar=dict(
            title      = "Mean<br>LOS (Days)",
            tickfont   = dict(color=DARK_GREY, size=11),
            thickness  = 14,
            len        = 0.8
        ),
        margin     = dict(l=220, r=20, t=50, b=80),
        hoverlabel = dict(bgcolor=NHS_DARK_BLUE,
                          font_size=12, font_color=WHITE)
    )

    fig.update_xaxes(
        tickmode="linear", dtick=1, tickangle=90,
        tickfont=dict(color=DARK_GREY, size=11),
        title_font=dict(color=NHS_DARK_BLUE),
        gridcolor=LIGHT_GREY, linecolor=LIGHT_GREY
    )

    fig.update_yaxes(
        title_text="",
        tickfont=dict(color=DARK_GREY, size=11),
        gridcolor=LIGHT_GREY, linecolor=LIGHT_GREY,
        ticksuffix="  ", ticklabelstandoff=10
    )
 
    return fig
    

 # Line chart
 # Found to the right of the heatmap. Dynamically updates to reflect which diagnosis code the user is currently looking at.

@app.callback(
    Output("line-chart", "figure"),
    Input("heatmap", "hoverData"),
    Input("category-dropdown", "value"),
    Input("sort-method", "value")
)
def update_line_chart(hoverData, selected_category, sort_method):
    """
    Dash callback that renders a line chart showing the mean LOS trend over time
    for a single diagnosis. The condition displayed is determined by which heatmap
    cell the user is hovering over; if no hover is active, the first condition in
    the current top 5 is shown by default.
    """
 
    filtered   = df[df["category"] == selected_category]
    top5_codes = analytics(filtered, sort_method)
    top5_df    = filtered[filtered["diagnosis_code"].isin(top5_codes)]
 
    code_to_label = (top5_df.groupby("diagnosis_code")["description"]
                     .first()
                     .str.slice(0, 30)
                     .add("...")
                     .to_dict())
    label_to_code = {v: k for k, v in code_to_label.items()}
 
    if hoverData is None:
        condition_code = top5_codes[0]
    else:
        hovered_label  = hoverData["points"][0]["y"]
        condition_code = label_to_code.get(hovered_label, top5_codes[0])
 
    desc_matches    = filtered[filtered["diagnosis_code"] == condition_code]["description"]
    condition_title = desc_matches.iloc[0] if len(desc_matches) > 0 else condition_code
 
    condition_df = filtered[filtered["diagnosis_code"] == condition_code]
    trend = (condition_df.groupby("year")["mean_length_stay"]
             .mean().reset_index())
 
    fig = px.line(
        trend, x="year", y="mean_length_stay",
        title=condition_title,
        labels={
            "mean_length_stay": "Mean Length of Stay (Days)",
            "year": "Year"
        },
        markers=True
    )
 
    fig.update_traces(
        line=dict(color=NHS_BLUE, width=2.5),
        marker=dict(size=6, color=NHS_DARK_BLUE,
                    line=dict(width=1, color=WHITE)),
        hovertemplate="Year: %{x}<br>Mean LOS: %{y:.1f} days<extra></extra>"
    )
 
    fig.update_layout(
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="Inter, Arial, sans-serif",
                  color=DARK_GREY, size=12),
        title=dict(font=dict(size=13, color=NHS_DARK_BLUE), x=0.01),
        margin=dict(l=20, r=20, t=50, b=20),
        hoverlabel=dict(bgcolor=NHS_DARK_BLUE,
                        font_size=12, font_color=WHITE)
    )
 
    fig.update_xaxes(
        tickmode="linear", dtick=2, tickangle=45,
        tickfont=dict(color=DARK_GREY, size=11),
        title_font=dict(color=NHS_DARK_BLUE),
        gridcolor=LIGHT_GREY, linecolor=LIGHT_GREY,
        zeroline=False
    )
 
    fig.update_yaxes(
        tickfont=dict(color=DARK_GREY, size=11),
        title_font=dict(color=NHS_DARK_BLUE),
        gridcolor=LIGHT_GREY, linecolor=LIGHT_GREY,
        zeroline=False
    )
 
    return fig
 
 # Scatter plot
 # Visualises the change in trends for which ever category and filter method have been selected
 
@app.callback(
    Output("scatter-plot", "figure"),
    Input("category-dropdown", "value"),
    Input("sort-method", "value")
)
def update_scatter(selected_category, sort_method):
    """
    Dash callback that renders an animated scatter plot of total admissions vs
    mean LOS over time for the selected conditions. Each bubble represents a
    condition in a given year, with bubble size encoding admission volume.
    The animation frame is set to year, allowing playback across 1998–2023.
    """
 
    filtered   = df[df["category"] == selected_category]
    top5_names = analytics(filtered, sort_method)
    top5_df    = filtered[filtered["diagnosis_code"].isin(top5_names)]
 
    scatter_df = (top5_df.groupby(["year", "diagnosis_code", "description"])
                  .agg(
                      total_admissions=("admissions",       "sum"),
                      mean_los        =("mean_length_stay", "mean")
                  ).reset_index())
 
    fig = px.scatter(
        scatter_df,
        x               = "total_admissions",
        y               = "mean_los",
        color           = "description",
        animation_frame = "year",
        size            = "total_admissions",
        size_max        = 40,
        title           = f"Admissions Volume vs Mean LOS Over Time — {selected_category}",
        range_x         = [0, scatter_df["total_admissions"].max() * 1.4],
        range_y         = [0, scatter_df["mean_los"].max() * 1.2],
        labels={
            "total_admissions" : "Total Admissions",
            "mean_los"         : "Mean Length of Stay (Days)",
            "description"      : "Condition",
            "year"             : "Year"
        },
        color_discrete_sequence=[
            NHS_BLUE, NHS_DARK_BLUE, NHS_LIGHT_BLUE, YELLOW, RED
        ]
    )
 
    fig.update_layout(
        plot_bgcolor  = WHITE,
        paper_bgcolor = WHITE,
        font=dict(family="Inter, Arial, sans-serif",
                  color=DARK_GREY, size=12),
        title=dict(font=dict(size=15, color=NHS_DARK_BLUE), x=0.01),
        legend=dict(
            title       = "Condition",
            font        = dict(size=11, color=DARK_GREY),
            bgcolor     = LIGHT_GREY,
            bordercolor = LIGHT_GREY
        ),
        margin     = dict(l=20, r=20, t=60, b=20),
        hoverlabel = dict(bgcolor=NHS_DARK_BLUE,
                          font_size=12, font_color=WHITE)
    )
 
    fig.update_xaxes(
        tickfont   = dict(color=DARK_GREY, size=11),
        title_font = dict(color=NHS_DARK_BLUE),
        gridcolor  = LIGHT_GREY, linecolor=LIGHT_GREY
    )
 
    fig.update_yaxes(
        range      = [0, df["mean_length_stay"].quantile(0.95)],
        tickfont   = dict(color=DARK_GREY, size=11),
        title_font = dict(color=NHS_DARK_BLUE),
        gridcolor  = LIGHT_GREY, linecolor=LIGHT_GREY
    )
 
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Admissions: %{x:,.0f}<br>"
            "Mean LOS: %{y:.1f} days"
            "<extra></extra>"
        ),
        customdata=scatter_df[["description"]].values
    )
 
    return fig
 
 
if __name__ == "__main__":
    app.run(debug=True)