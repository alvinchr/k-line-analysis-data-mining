import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==================================================
# 50 STOCKS
# ==================================================

stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AVGO", "ORCL", "ADBE",
    "JPM", "V", "MA", "BAC", "WFC",
    "COST", "WMT", "HD", "MCD", "KO",
    "PEP", "DIS", "NFLX", "CRM", "AMD",
    "INTC", "QCOM", "TXN", "IBM", "CSCO",
    "CAT", "BA", "GE", "UPS", "HON",
    "NKE", "SBUX", "T", "VZ", "XOM",
    "CVX", "COP", "JNJ", "PFE", "MRK",
    "ABBV", "UNH", "LLY", "AMGN", "GS"
]


# ==================================================
# 10 STOCKS FOR FULL ASSIGNMENT ANALYSIS
# ==================================================

analysis_stocks = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    "NVDA",
    "AMD",
    "TSLA",
    "JPM",
    "WMT"
]


# ==================================================
# GLOBAL DATA
# ==================================================

current_data = None
pattern_candidates = None


# ==================================================
# DOWNLOAD DATA
# ==================================================

def load_data():

    stock = stock_box.get()

    try:

        status_label.config(
            text=f"Downloading {stock}...",
            fg="#2563EB"
        )

        window.update()

        data = yf.download(
            stock,
            start="2018-01-01",
            end="2027-01-01",
            auto_adjust=True,
            progress=False
        )

        if data.empty:

            messagebox.showerror(
                "Error",
                "No data found."
            )

            return

        # Handle yfinance MultiIndex
        if isinstance(data.columns, pd.MultiIndex):

            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        global current_data

        current_data = data

        # Clear old OHLCV table
        for item in table.get_children():

            table.delete(item)

        # Show latest 30 rows
        display_data = data.tail(30)

        for _, row in display_data.iterrows():

            table.insert(
                "",
                "end",
                values=(
                    row["Date"].strftime("%Y-%m-%d"),
                    f"{row['Open']:.2f}",
                    f"{row['High']:.2f}",
                    f"{row['Low']:.2f}",
                    f"{row['Close']:.2f}",
                    f"{row['Volume']:,.0f}"
                )
            )

        status_label.config(
            text=f"{stock}: {len(data)} trading days loaded",
            fg="#16A34A"
        )

        # Draw chart
        draw_chart(
            data,
            stock
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            str(e)
        )

        status_label.config(
            text="Download failed",
            fg="#DC2626"
        )


# ==================================================
# K-LINE FEATURE EXTRACTION
# ==================================================

def calculate_features(data):

    data = data.copy()

    # K-line total range
    data["range"] = (
        data["High"]
        - data["Low"]
    )

    # Candle body
    data["body"] = abs(
        data["Close"]
        - data["Open"]
    )

    # Upper shadow
    data["upper_shadow"] = (
        data["High"]
        - data[["Open", "Close"]].max(axis=1)
    )

    # Lower shadow
    data["lower_shadow"] = (
        data[["Open", "Close"]].min(axis=1)
        - data["Low"]
    )

    # Avoid division by zero
    safe_range = data["range"].replace(
        0,
        np.nan
    )

    # Normalized candle features
    data["body_ratio"] = (
        data["body"]
        / safe_range
    )

    data["upper_ratio"] = (
        data["upper_shadow"]
        / safe_range
    )

    data["lower_ratio"] = (
        data["lower_shadow"]
        / safe_range
    )

    # Candle direction
    data["direction"] = np.where(
        data["Close"] >= data["Open"],
        1,
        -1
    )

    # Volume relative to 20-day average
    volume_average = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["volume_ratio"] = (
        data["Volume"]
        / volume_average
    )

    return data


# ==================================================
# FIND +5% / -5% PATTERNS
# ==================================================

def find_patterns():

    global current_data
    global pattern_candidates

    if current_data is None:

        messagebox.showwarning(
            "No Data",
            "Please load a stock first."
        )

        return

    try:

        status_label.config(
            text="Analyzing 2018-2025 training data...",
            fg="#2563EB"
        )

        window.update()

        data = calculate_features(
            current_data
        )

        # ==========================================
        # TRAINING DATA ONLY
        # ==========================================

        train = data[
            (data["Date"] >= "2018-01-01") &
            (data["Date"] <= "2025-12-31")
        ].copy()

        # ==========================================
        # 3-DAY FUTURE RETURN
        # ==========================================

        train["future_return_3d"] = (
            train["Close"].shift(-3)
            / train["Close"]
            - 1
        )

        # ==========================================
        # SIGNAL
        #
        # +1 = future price rises > 5%
        # -1 = future price falls > 5%
        #  0 = neither
        # ==========================================

        train["signal"] = 0

        train.loc[
            train["future_return_3d"] > 0.05,
            "signal"
        ] = 1

        train.loc[
            train["future_return_3d"] < -0.05,
            "signal"
        ] = -1

        # ==========================================
        # REMOVE INCOMPLETE ROWS
        # ==========================================

        train = train.dropna(
            subset=[
                "future_return_3d",
                "body_ratio",
                "upper_ratio",
                "lower_ratio",
                "volume_ratio"
            ]
        )

        # ==========================================
        # KEEP ONLY SUCCESS CASES
        # ==========================================

        candidates = train[
            train["signal"] != 0
        ].copy()

        # Keep important features
        candidates = candidates[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "body_ratio",
                "upper_ratio",
                "lower_ratio",
                "volume_ratio",
                "direction",
                "future_return_3d",
                "signal"
            ]
        ]

        pattern_candidates = candidates

        # Count candidates
        bullish_count = len(
            candidates[
                candidates["signal"] == 1
            ]
        )

        bearish_count = len(
            candidates[
                candidates["signal"] == -1
            ]
        )

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================

        for item in pattern_table.get_children():

            pattern_table.delete(item)

        # Show latest 100 candidates
        display_candidates = candidates.tail(100)

        for _, row in display_candidates.iterrows():

            if row["signal"] == 1:

                signal_text = "BULLISH"

            else:

                signal_text = "BEARISH"

            pattern_table.insert(
                "",
                "end",
                values=(
                    row["Date"].strftime("%Y-%m-%d"),
                    signal_text,
                    f"{row['body_ratio']:.3f}",
                    f"{row['upper_ratio']:.3f}",
                    f"{row['lower_ratio']:.3f}",
                    f"{row['volume_ratio']:.2f}",
                    f"{row['future_return_3d'] * 100:.2f}%"
                )
            )

        status_label.config(
            text=(
                f"Training complete | "
                f"Bullish: {bullish_count} | "
                f"Bearish: {bearish_count}"
            ),
            fg="#16A34A"
        )

    except Exception as e:

        messagebox.showerror(
            "Analysis Error",
            str(e)
        )

        status_label.config(
            text="Analysis failed",
            fg="#DC2626"
        )


# ==================================================
# RUN ANALYSIS ON 10 SELECTED STOCKS
# ==================================================

def run_10_stocks():

    results = []

    # ==============================================
    # CLEAR OLD RESULTS
    # ==============================================

    for item in stock_result_table.get_children():

        stock_result_table.delete(item)

    # ==============================================
    # SHOW ALL STOCKS AS WAITING
    # ==============================================

    for stock in analysis_stocks:

        stock_result_table.insert(
            "",
            "end",
            iid=stock,
            values=(
                stock,
                "2018-2025",
                "-",
                "-",
                "WAITING"
            )
        )

    status_label.config(
        text="Running analysis on 10 stocks...",
        fg="#2563EB"
    )

    window.update()

    # ==============================================
    # ANALYZE EACH STOCK
    # ==============================================

    for stock in analysis_stocks:

        try:

            # --------------------------------------
            # SHOW RUNNING
            # --------------------------------------

            stock_result_table.item(
                stock,
                values=(
                    stock,
                    "2018-2025",
                    "-",
                    "-",
                    "RUNNING"
                )
            )

            status_label.config(
                text=f"Analyzing {stock}...",
                fg="#2563EB"
            )

            window.update()

            # --------------------------------------
            # DOWNLOAD DATA
            # --------------------------------------

            data = yf.download(
                stock,
                start="2018-01-01",
                end="2027-01-01",
                auto_adjust=True,
                progress=False
            )

            if data.empty:

                stock_result_table.item(
                    stock,
                    values=(
                        stock,
                        "2018-2025",
                        "-",
                        "-",
                        "FAILED"
                    )
                )

                window.update()

                continue

            # --------------------------------------
            # HANDLE MULTIINDEX
            # --------------------------------------

            if isinstance(data.columns, pd.MultiIndex):

                data.columns = (
                    data.columns
                    .get_level_values(0)
                )

            data = data.reset_index()

            # --------------------------------------
            # CALCULATE FEATURES
            # --------------------------------------

            data = calculate_features(
                data
            )

            # --------------------------------------
            # TRAINING DATA
            # --------------------------------------

            train = data[
                (data["Date"] >= "2018-01-01") &
                (data["Date"] <= "2025-12-31")
            ].copy()

            # --------------------------------------
            # 3-DAY FUTURE RETURN
            # --------------------------------------

            train["future_return_3d"] = (
                train["Close"].shift(-3)
                / train["Close"]
                - 1
            )

            # --------------------------------------
            # SIGNAL
            # --------------------------------------

            train["signal"] = 0

            # Bullish
            train.loc[
                train["future_return_3d"] > 0.05,
                "signal"
            ] = 1

            # Bearish
            train.loc[
                train["future_return_3d"] < -0.05,
                "signal"
            ] = -1

            # --------------------------------------
            # REMOVE INCOMPLETE ROWS
            # --------------------------------------

            train = train.dropna(
                subset=[
                    "future_return_3d",
                    "body_ratio",
                    "upper_ratio",
                    "lower_ratio",
                    "volume_ratio"
                ]
            )

            # --------------------------------------
            # COUNT CANDIDATES
            # --------------------------------------

            bullish = len(
                train[
                    train["signal"] == 1
                ]
            )

            bearish = len(
                train[
                    train["signal"] == -1
                ]
            )

            # Save result
            results.append(
                (
                    stock,
                    bullish,
                    bearish,
                    len(train)
                )
            )

            # --------------------------------------
            # SHOW COMPLETE
            # --------------------------------------

            stock_result_table.item(
                stock,
                values=(
                    stock,
                    "2018-2025",
                    bullish,
                    bearish,
                    "COMPLETE"
                )
            )

            window.update()

        except Exception as e:

            print(
                f"{stock} failed: {e}"
            )

            stock_result_table.item(
                stock,
                values=(
                    stock,
                    "2018-2025",
                    "-",
                    "-",
                    "FAILED"
                )
            )

            window.update()

    # ==============================================
    # FINISHED
    # ==============================================

    status_label.config(
        text=(
            f"10-stock analysis completed: "
            f"{len(results)} stocks"
        ),
        fg="#16A34A"
    )

    window.update()


# ==================================================
# WEIGHTED SIMILARITY
# ==================================================

def pattern_similarity(
    pattern_a,
    pattern_b
):

    # ==========================================
    # FEATURES
    # ==========================================

    features = [
        "body_ratio",
        "upper_ratio",
        "lower_ratio",
        "volume_ratio"
    ]

    # ==========================================
    # INITIAL WEIGHTS
    # ==========================================

    weights = {

        "body_ratio": 0.30,

        "upper_ratio": 0.20,

        "lower_ratio": 0.30,

        "volume_ratio": 0.20
    }

    # ==========================================
    # WEIGHTED EUCLIDEAN DISTANCE
    # ==========================================

    distance = 0

    for feature in features:

        difference = (
            pattern_a[feature]
            - pattern_b[feature]
        )

        distance += (
            weights[feature]
            * difference ** 2
        )

    distance = np.sqrt(
        distance
    )

    # ==========================================
    # CONVERT DISTANCE TO SIMILARITY
    # ==========================================

    similarity = (
        1
        / (1 + distance)
    )

    return similarity


# ==================================================
# TEST SIMILARITY BUTTON
# ==================================================

def test_similarity():

    global pattern_candidates

    if pattern_candidates is None:

        messagebox.showwarning(
            "No Patterns",
            "Run FIND PATTERNS first."
        )

        return

    if len(pattern_candidates) < 2:

        messagebox.showwarning(
            "Not Enough Data",
            "Not enough patterns found."
        )

        return

    # Take first two candidate patterns
    pattern_a = pattern_candidates.iloc[0]

    pattern_b = pattern_candidates.iloc[1]

    similarity = pattern_similarity(
        pattern_a,
        pattern_b
    )

    messagebox.showinfo(
        "Pattern Similarity",
        (
            f"Pattern A: "
            f"{pattern_a['Date'].strftime('%Y-%m-%d')}\n\n"

            f"Pattern B: "
            f"{pattern_b['Date'].strftime('%Y-%m-%d')}\n\n"

            f"Similarity: "
            f"{similarity:.4f}\n\n"

            f"Similarity percentage: "
            f"{similarity * 100:.2f}%"
        )
    )


# ==================================================
# DRAW K-LINE CHART
# ==================================================

def draw_chart(
    data,
    stock
):

    # Remove previous chart
    for widget in chart_frame.winfo_children():

        widget.destroy()

    # Last 80 trading days
    recent = data.tail(80).copy()

    fig, ax = plt.subplots(
        figsize=(9, 2.2),
        dpi=100
    )

    # ==========================================
    # DRAW EACH CANDLE
    # ==========================================

    for i, row in recent.iterrows():

        x = list(
            recent.index
        ).index(i)

        open_price = row["Open"]

        high_price = row["High"]

        low_price = row["Low"]

        close_price = row["Close"]

        # Green = up
        # Red = down

        if close_price >= open_price:

            color = "green"

        else:

            color = "red"

        # --------------------------------------
        # Wick
        # --------------------------------------

        ax.plot(
            [x, x],
            [
                low_price,
                high_price
            ],
            color=color,
            linewidth=1
        )

        # --------------------------------------
        # Candle body
        # --------------------------------------

        bottom = min(
            open_price,
            close_price
        )

        height = abs(
            close_price
            - open_price
        )

        ax.bar(
            x,
            height,
            bottom=bottom,
            width=0.6,
            color=color
        )

    # ==========================================
    # CHART LABELS
    # ==========================================

    ax.set_title(
        f"{stock} K-Line Chart",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Trading Days"
    )

    ax.set_ylabel(
        "Price"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    # ==========================================
    # PUT CHART INTO TKINTER
    # ==========================================

    canvas = FigureCanvasTkAgg(
        fig,
        master=chart_frame
    )

    canvas.draw()

    canvas.get_tk_widget().pack(
        fill="both",
        expand=True
    )


# ==================================================
# MAIN WINDOW
# ==================================================

window = tk.Tk()

window.title(
    "K-Line Pattern Data Mining"
)

window.geometry(
    "1150x800"
)

window.configure(
    bg="#F4F7FB"
)


# ==================================================
# HEADER
# ==================================================

header = tk.Frame(
    window,
    bg="#243B53",
    height=90
)

header.pack(
    fill="x"
)

tk.Label(
    header,
    text="K-Line Pattern Data Mining",
    font=("Arial", 24, "bold"),
    bg="#243B53",
    fg="white"
).pack(
    pady=(8, 2)
)

tk.Label(
    header,
    text=(
        "Stock Data • "
        "K-Line Visualization • "
        "Pattern Analysis"
    ),
    font=("Arial", 10),
    bg="#243B53",
    fg="#D9E8FF"
).pack()


# ==================================================
# CONTROL AREA
# ==================================================

control = tk.Frame(
    window,
    bg="white",
    padx=20,
    pady=7
)

control.pack(
    fill="x",
    padx=20,
    pady=7
)


tk.Label(
    control,
    text="Select Stock:",
    font=("Arial", 11, "bold"),
    bg="white",
    fg="#243B53"
).pack(
    side="left"
)


stock_box = ttk.Combobox(
    control,
    values=stocks,
    width=15,
    state="readonly",
    font=("Arial", 11)
)

stock_box.set(
    "AAPL"
)

stock_box.pack(
    side="left",
    padx=10
)


# LOAD DATA

tk.Button(
    control,
    text="LOAD DATA",
    command=load_data,
    font=("Arial", 11, "bold"),
    bg="#2563EB",
    fg="white",
    padx=15,
    pady=7,
    relief="flat"
).pack(
    side="left"
)


# FIND PATTERNS

tk.Button(
    control,
    text="FIND PATTERNS",
    command=find_patterns,
    font=("Arial", 11, "bold"),
    bg="#16A34A",
    fg="white",
    padx=15,
    pady=7,
    relief="flat"
).pack(
    side="left",
    padx=8
)


# TEST SIMILARITY

tk.Button(
    control,
    text="TEST SIMILARITY",
    command=test_similarity,
    font=("Arial", 11, "bold"),
    bg="#9333EA",
    fg="white",
    padx=15,
    pady=7,
    relief="flat"
).pack(
    side="left"
)


# STATUS

status_label = tk.Label(
    control,
    text="Select a stock and click LOAD DATA",
    bg="white",
    fg="#64748B",
    font=("Arial", 10)
)

status_label.pack(
    side="left",
    padx=15
)


# ==================================================
# HISTORICAL OHLCV DATA
# ==================================================

tk.Label(
    window,
    text="Historical OHLCV Data",
    font=("Arial", 15, "bold"),
    bg="#F4F7FB",
    fg="#243B53"
).pack(
    anchor="w",
    padx=25
)


table_frame = tk.Frame(
    window,
    bg="white"
)

table_frame.pack(
    fill="x",
    padx=25,
    pady=8
)


columns = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
)


table = ttk.Treeview(
    table_frame,
    columns=columns,
    show="headings",
    height=3
)


for column in columns:

    table.heading(
        column,
        text=column
    )

    table.column(
        column,
        width=150,
        anchor="center"
    )


table.pack(
    side="left",
    fill="x",
    expand=True
)


scrollbar = ttk.Scrollbar(
    table_frame,
    orient="vertical",
    command=table.yview
)

scrollbar.pack(
    side="right",
    fill="y"
)

table.configure(
    yscrollcommand=scrollbar.set
)


# ==================================================
# DETECTED TRAINING PATTERNS
# ==================================================

tk.Label(
    window,
    text="Detected Training Patterns",
    font=("Arial", 15, "bold"),
    bg="#F4F7FB",
    fg="#243B53"
).pack(
    anchor="w",
    padx=25,
    pady=(8, 0)
)


pattern_frame = tk.Frame(
    window,
    bg="white"
)

pattern_frame.pack(
    fill="x",
    padx=25,
    pady=8
)


pattern_columns = (
    "Date",
    "Signal",
    "Body",
    "Upper Shadow",
    "Lower Shadow",
    "Volume Ratio",
    "3-Day Return"
)


pattern_table = ttk.Treeview(
    pattern_frame,
    columns=pattern_columns,
    show="headings",
    height=3
)


for column in pattern_columns:

    pattern_table.heading(
        column,
        text=column
    )

    pattern_table.column(
        column,
        width=140,
        anchor="center"
    )


pattern_table.pack(
    side="left",
    fill="x",
    expand=True
)


pattern_scrollbar = ttk.Scrollbar(
    pattern_frame,
    orient="vertical",
    command=pattern_table.yview
)

pattern_scrollbar.pack(
    side="right",
    fill="y"
)

pattern_table.configure(
    yscrollcommand=pattern_scrollbar.set
)


# ==================================================
# 10-STOCK ASSIGNMENT ANALYSIS
# ==================================================

tk.Label(
    window,
    text="10-Stock Assignment Analysis",
    font=("Arial", 15, "bold"),
    bg="#F4F7FB",
    fg="#243B53"
).pack(
    anchor="w",
    padx=25,
    pady=(8, 0)
)


# ------------------------------------------
# 10-STOCK CONTROL AREA
# ------------------------------------------

stock_analysis_control = tk.Frame(
    window,
    bg="white",
    padx=15,
    pady=10
)

stock_analysis_control.pack(
    fill="x",
    padx=25,
    pady=5
)


tk.Label(
    stock_analysis_control,
    text="Stocks:",
    font=("Arial", 10, "bold"),
    bg="white",
    fg="#243B53"
).pack(
    side="left"
)


tk.Label(
    stock_analysis_control,
    text=(
        "AAPL • MSFT • GOOGL • META • AMZN • "
        "NVDA • AMD • TSLA • JPM • WMT"
    ),
    font=("Arial", 10),
    bg="white",
    fg="#475569"
).pack(
    side="left",
    padx=7
)


# ------------------------------------------
# RUN 10-STOCK ANALYSIS BUTTON
# ------------------------------------------

tk.Button(
    stock_analysis_control,
    text="RUN 10-STOCK ANALYSIS",
    command=run_10_stocks,
    font=("Arial", 10, "bold"),
    bg="#EA580C",
    fg="white",
    padx=15,
    pady=7,
    relief="flat"
).pack(
    side="right"
)


# ------------------------------------------
# 10-STOCK RESULT TABLE
# ------------------------------------------

stock_result_frame = tk.Frame(window, bg="white")
stock_result_frame.pack(fill="x", padx=25, pady=5)

stock_result_title = tk.Label(
    stock_result_frame,
    text="10-Stock Assignment Analysis",
    font=("Arial", 12, "bold"),
    bg="white"
)
stock_result_title.pack(anchor="w")

# Frame for table + scrollbar
stock_table_container = tk.Frame(stock_result_frame, bg="white")
stock_table_container.pack(fill="x")

stock_result_columns = (
    "Stock",
    "Status",
    "Bullish",
    "Bearish"
)

stock_result_table = ttk.Treeview(
    stock_table_container,
    columns=stock_result_columns,
    show="headings",
    height=3
)

# Scrollbar
stock_scrollbar = ttk.Scrollbar(
    stock_table_container,
    orient="vertical",
    command=stock_result_table.yview
)

stock_result_table.configure(
    yscrollcommand=stock_scrollbar.set
)

# Headings
for col in stock_result_columns:
    stock_result_table.heading(col, text=col)

# Column widths
stock_result_table.column("Stock", width=100)
stock_result_table.column("Status", width=150)
stock_result_table.column("Bullish", width=100)
stock_result_table.column("Bearish", width=100)

# Put table + scrollbar side by side
stock_result_table.pack(
    side="left",
    fill="x",
    expand=True
)

stock_scrollbar.pack(
    side="right",
    fill="y"
)

# ==================================================
# K-LINE CHART
# ==================================================

tk.Label(
    window,
    text="K-Line Chart",
    font=("Arial", 15, "bold"),
    bg="#F4F7FB",
    fg="#243B53"
).pack(
    anchor="w",
    padx=25,
    pady=(8, 0)
)


# FIXED HEIGHT
# This prevents the chart from pushing
# the 10-stock section away.

chart_frame = tk.Frame(
    window,
    bg="white",
    height=200
)

chart_frame.pack(
    fill="x",
    padx=25,
    pady=5
)

chart_frame.pack_propagate(
    False
)


# ==================================================
# START PROGRAM
# ==================================================

window.mainloop()