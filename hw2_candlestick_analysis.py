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

# Store complete results from 10-stock analysis
ten_stock_data = {}

# Store Top 10 bullish / bearish cases
top_bullish_results = []
top_bearish_results = []


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

    global ten_stock_data
    global top_bullish_results
    global top_bearish_results

    # Reset previous results
    ten_stock_data = {}
    top_bullish_results = []
    top_bearish_results = []

    # ==============================================
    # CLEAR OLD RESULTS
    # ==============================================

    for item in stock_result_table.get_children():

        stock_result_table.delete(item)

    for item in bullish_top_table.get_children():

        bullish_top_table.delete(item)

    for item in bearish_top_table.get_children():

        bearish_top_table.delete(item)

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
                "WAITING",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-"
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

            stock_result_table.item(
                stock,
                values=(
                    stock,
                    "RUNNING",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-"
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
                        "FAILED",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-"
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

            data = calculate_features(data)

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

            train.loc[
                train["future_return_3d"] > 0.05,
                "signal"
            ] = 1

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
            # KEEP ONLY BULLISH / BEARISH EVENTS
            # --------------------------------------

            candidates = train[
                train["signal"] != 0
            ].copy()

            # --------------------------------------
            # COUNT
            # --------------------------------------

            bullish_candidates = candidates[
                candidates["signal"] == 1
            ].copy()

            bearish_candidates = candidates[
                candidates["signal"] == -1
            ].copy()

            bullish_count = len(
                bullish_candidates
            )

            bearish_count = len(
                bearish_candidates
            )

            # --------------------------------------
            # FIND HIGHEST BULLISH CASE
            # --------------------------------------

            if not bullish_candidates.empty:

                bullish_idx = (
                    bullish_candidates[
                        "future_return_3d"
                    ].idxmax()
                )

                highest_bullish = (
                    bullish_candidates.loc[
                        bullish_idx
                    ]
                )

            else:

                highest_bullish = None

            # --------------------------------------
            # FIND HIGHEST BEARISH CASE
            # --------------------------------------

            if not bearish_candidates.empty:

                bearish_idx = (
                    bearish_candidates[
                        "future_return_3d"
                    ].idxmin()
                )

                highest_bearish = (
                    bearish_candidates.loc[
                        bearish_idx
                    ]
                )

            else:

                highest_bearish = None

            # --------------------------------------
            # SAVE ALL DATA
            # --------------------------------------

            ten_stock_data[stock] = {
                "data": data,
                "train": train,
                "candidates": candidates,
                "bullish": bullish_candidates,
                "bearish": bearish_candidates,
                "highest_bullish": highest_bullish,
                "highest_bearish": highest_bearish
            }

            # --------------------------------------
            # VALUES FOR SUMMARY TABLE
            # --------------------------------------

            if highest_bullish is not None:

                bullish_return = (
                    highest_bullish[
                        "future_return_3d"
                    ] * 100
                )

                bullish_date = (
                    highest_bullish["Date"]
                    .strftime("%Y-%m-%d")
                )

            else:

                bullish_return = 0
                bullish_date = "-"

            if highest_bearish is not None:

                bearish_return = (
                    highest_bearish[
                        "future_return_3d"
                    ] * 100
                )

                bearish_date = (
                    highest_bearish["Date"]
                    .strftime("%Y-%m-%d")
                )

            else:

                bearish_return = 0
                bearish_date = "-"

            # --------------------------------------
            # UPDATE SUMMARY TABLE
            # --------------------------------------

            stock_result_table.item(
                stock,
                values=(
                    stock,
                    "COMPLETE",
                    bullish_count,
                    bearish_count,
                    f"+{bullish_return:.2f}%",
                    bullish_date,
                    f"{bearish_return:.2f}%",
                    bearish_date
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
                    "FAILED",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-"
                )
            )

            window.update()

    # ==============================================
    # BUILD TOP 10 RESULTS
    # ==============================================

    build_top_10_results()

    # ==============================================
    # FINISHED
    # ==============================================

    status_label.config(
        text=(
            f"10-stock analysis completed | "
            f"{len(ten_stock_data)} stocks analyzed | "
            f"Top 10 results generated"
        ),
        fg="#16A34A"
    )

    window.update()


# ==================================================
# BUILD TOP 10 BULLISH / BEARISH
# ==================================================

def build_top_10_results():

    global top_bullish_results
    global top_bearish_results

    bullish_all = []
    bearish_all = []

    # ==============================================
    # COLLECT RESULTS FROM ALL 10 STOCKS
    # ==============================================

    for stock, result in ten_stock_data.items():

        bullish = result["bullish"]
        bearish = result["bearish"]

        # ------------------------------------------
        # BULLISH
        # ------------------------------------------

        for _, row in bullish.iterrows():

            bullish_all.append({
                "stock": stock,
                "date": row["Date"],
                "return": row["future_return_3d"],
                "body_ratio": row["body_ratio"],
                "upper_ratio": row["upper_ratio"],
                "lower_ratio": row["lower_ratio"],
                "volume_ratio": row["volume_ratio"],
                "close": row["Close"]
            })

        # ------------------------------------------
        # BEARISH
        # ------------------------------------------

        for _, row in bearish.iterrows():

            bearish_all.append({
                "stock": stock,
                "date": row["Date"],
                "return": row["future_return_3d"],
                "body_ratio": row["body_ratio"],
                "upper_ratio": row["upper_ratio"],
                "lower_ratio": row["lower_ratio"],
                "volume_ratio": row["volume_ratio"],
                "close": row["Close"]
            })

    # ==============================================
    # SORT
    # ==============================================

    bullish_all.sort(
        key=lambda x: x["return"],
        reverse=True
    )

    bearish_all.sort(
        key=lambda x: x["return"]
    )

    # ==============================================
    # KEEP TOP 10
    # ==============================================

    top_bullish_results = bullish_all[:10]

    top_bearish_results = bearish_all[:10]

    # ==============================================
    # DISPLAY TOP 10 BULLISH
    # ==============================================

    for item in bullish_top_table.get_children():

        bullish_top_table.delete(item)

    for rank, result in enumerate(
        top_bullish_results,
        start=1
    ):

        bullish_top_table.insert(
            "",
            "end",
            values=(
                rank,
                result["stock"],
                result["date"].strftime("%Y-%m-%d"),
                f"+{result['return'] * 100:.2f}%",
                f"{result['body_ratio']:.3f}",
                f"{result['upper_ratio']:.3f}",
                f"{result['lower_ratio']:.3f}",
                f"{result['volume_ratio']:.2f}"
            )
        )

    # ==============================================
    # DISPLAY TOP 10 BEARISH
    # ==============================================

    for item in bearish_top_table.get_children():

        bearish_top_table.delete(item)

    for rank, result in enumerate(
        top_bearish_results,
        start=1
    ):

        bearish_top_table.insert(
            "",
            "end",
            values=(
                rank,
                result["stock"],
                result["date"].strftime("%Y-%m-%d"),
                f"{result['return'] * 100:.2f}%",
                f"{result['body_ratio']:.3f}",
                f"{result['upper_ratio']:.3f}",
                f"{result['lower_ratio']:.3f}",
                f"{result['volume_ratio']:.2f}"
            )
        )


# ==================================================
# SHOW TOP PATTERN DETAIL
# ==================================================

def show_selected_bullish(event=None):

    selected = bullish_top_table.selection()

    if not selected:
        return

    values = bullish_top_table.item(
        selected[0],
        "values"
    )

    messagebox.showinfo(
        "Bullish Pattern Detail",
        (
            f"Rank: {values[0]}\n"
            f"Stock: {values[1]}\n"
            f"Date: {values[2]}\n"
            f"3-Day Return: {values[3]}\n\n"
            f"Body Ratio: {values[4]}\n"
            f"Upper Ratio: {values[5]}\n"
            f"Lower Ratio: {values[6]}\n"
            f"Volume Ratio: {values[7]}"
        )
    )


def show_selected_bearish(event=None):

    selected = bearish_top_table.selection()

    if not selected:
        return

    values = bearish_top_table.item(
        selected[0],
        "values"
    )

    messagebox.showinfo(
        "Bearish Pattern Detail",
        (
            f"Rank: {values[0]}\n"
            f"Stock: {values[1]}\n"
            f"Date: {values[2]}\n"
            f"3-Day Return: {values[3]}\n\n"
            f"Body Ratio: {values[4]}\n"
            f"Upper Ratio: {values[5]}\n"
            f"Lower Ratio: {values[6]}\n"
            f"Volume Ratio: {values[7]}"
        )
    )


# ==================================================
# WEIGHTED SIMILARITY
# ==================================================

def pattern_similarity(
    pattern_a,
    pattern_b
):

    features = [
        "body_ratio",
        "upper_ratio",
        "lower_ratio",
        "volume_ratio"
    ]

    weights = {

        "body_ratio": 0.30,

        "upper_ratio": 0.20,

        "lower_ratio": 0.30,

        "volume_ratio": 0.20
    }

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

        if close_price >= open_price:

            color = "green"

        else:

            color = "red"

        # Wick
        ax.plot(
            [x, x],
            [
                low_price,
                high_price
            ],
            color=color,
            linewidth=1
        )

        # Candle body
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
    "1250x850"
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
# SCROLLABLE MAIN AREA
# ==================================================

# Outer frame
scroll_container = tk.Frame(
    window,
    bg="#F4F7FB"
)

scroll_container.pack(
    fill="both",
    expand=True
)


# Canvas
main_canvas = tk.Canvas(
    scroll_container,
    bg="#F4F7FB",
    highlightthickness=0
)

main_canvas.pack(
    side="left",
    fill="both",
    expand=True
)


# Vertical scrollbar
main_scrollbar = ttk.Scrollbar(
    scroll_container,
    orient="vertical",
    command=main_canvas.yview
)

main_scrollbar.pack(
    side="right",
    fill="y"
)


main_canvas.configure(
    yscrollcommand=main_scrollbar.set
)


# This frame contains EVERYTHING below the header
content_frame = tk.Frame(
    main_canvas,
    bg="#F4F7FB"
)


content_window = main_canvas.create_window(
    (0, 0),
    window=content_frame,
    anchor="nw"
)


# ==================================================
# UPDATE SCROLL REGION
# ==================================================

def update_scroll_region(event=None):

    main_canvas.configure(
        scrollregion=main_canvas.bbox("all")
    )


content_frame.bind(
    "<Configure>",
    update_scroll_region
)


# Make content frame same width as canvas
def resize_content(event):

    main_canvas.itemconfig(
        content_window,
        width=event.width
    )


main_canvas.bind(
    "<Configure>",
    resize_content
)


# ==================================================
# MOUSE WHEEL SCROLLING
# ==================================================

def mouse_wheel(event):

    main_canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )


main_canvas.bind_all(
    "<MouseWheel>",
    mouse_wheel
)


# ==================================================
# CONTROL AREA
# ==================================================

control = tk.Frame(
    content_frame,
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
    content_frame,
    text="Historical OHLCV Data",
    font=("Arial", 15, "bold"),
    bg="#F4F7FB",
    fg="#243B53"
).pack(
    anchor="w",
    padx=25
)


table_frame = tk.Frame(
    content_frame,
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
    content_frame,
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
    content_frame,
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
    content_frame,
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
# CONTROL AREA
# ------------------------------------------

stock_analysis_control = tk.Frame(
    content_frame,
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


# ==================================================
# 10-STOCK SUMMARY TABLE
# ==================================================

stock_result_frame = tk.Frame(
    content_frame,
    bg="white"
)

stock_result_frame.pack(
    fill="x",
    padx=25,
    pady=5
)


tk.Label(
    stock_result_frame,
    text="10-Stock Historical Summary",
    font=("Arial", 12, "bold"),
    bg="white"
).pack(
    anchor="w"
)


stock_table_container = tk.Frame(
    stock_result_frame,
    bg="white"
)

stock_table_container.pack(
    fill="x"
)


stock_result_columns = (
    "Stock",
    "Status",
    "Bullish",
    "Bearish",
    "Max Bullish",
    "Bullish Date",
    "Max Bearish",
    "Bearish Date"
)


stock_result_table = ttk.Treeview(
    stock_table_container,
    columns=stock_result_columns,
    show="headings",
    height=5
)


stock_scrollbar = ttk.Scrollbar(
    stock_table_container,
    orient="vertical",
    command=stock_result_table.yview
)

stock_result_table.configure(
    yscrollcommand=stock_scrollbar.set
)


for col in stock_result_columns:

    stock_result_table.heading(
        col,
        text=col
    )


stock_result_table.column(
    "Stock",
    width=75,
    anchor="center"
)

stock_result_table.column(
    "Status",
    width=90,
    anchor="center"
)

stock_result_table.column(
    "Bullish",
    width=75,
    anchor="center"
)

stock_result_table.column(
    "Bearish",
    width=75,
    anchor="center"
)

stock_result_table.column(
    "Max Bullish",
    width=100,
    anchor="center"
)

stock_result_table.column(
    "Bullish Date",
    width=110,
    anchor="center"
)

stock_result_table.column(
    "Max Bearish",
    width=100,
    anchor="center"
)

stock_result_table.column(
    "Bearish Date",
    width=110,
    anchor="center"
)


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
# TOP 10 BULLISH
# ==================================================

tk.Label(
    content_frame,
    text="Top 10 Bullish Historical Patterns",
    font=("Arial", 14, "bold"),
    bg="#F4F7FB",
    fg="#15803D"
).pack(
    anchor="w",
    padx=25,
    pady=(8, 0)
)


bullish_frame = tk.Frame(
    content_frame,
    bg="white"
)

bullish_frame.pack(
    fill="x",
    padx=25,
    pady=5
)


bullish_columns = (
    "Rank",
    "Stock",
    "Date",
    "3-Day Return",
    "Body Ratio",
    "Upper Ratio",
    "Lower Ratio",
    "Volume Ratio"
)


bullish_top_table = ttk.Treeview(
    bullish_frame,
    columns=bullish_columns,
    show="headings",
    height=5
)


for column in bullish_columns:

    bullish_top_table.heading(
        column,
        text=column
    )

    bullish_top_table.column(
        column,
        width=125,
        anchor="center"
    )


bullish_top_table.pack(
    fill="x",
    expand=True
)


bullish_top_table.bind(
    "<Double-1>",
    show_selected_bullish
)


# ==================================================
# TOP 10 BEARISH
# ==================================================

tk.Label(
    content_frame,
    text="Top 10 Bearish Historical Patterns",
    font=("Arial", 14, "bold"),
    bg="#F4F7FB",
    fg="#B91C1C"
).pack(
    anchor="w",
    padx=25,
    pady=(8, 0)
)


bearish_frame = tk.Frame(
    content_frame,
    bg="white"
)

bearish_frame.pack(
    fill="x",
    padx=25,
    pady=5
)


bearish_columns = (
    "Rank",
    "Stock",
    "Date",
    "3-Day Return",
    "Body Ratio",
    "Upper Ratio",
    "Lower Ratio",
    "Volume Ratio"
)


bearish_top_table = ttk.Treeview(
    bearish_frame,
    columns=bearish_columns,
    show="headings",
    height=5
)


for column in bearish_columns:

    bearish_top_table.heading(
        column,
        text=column
    )

    bearish_top_table.column(
        column,
        width=125,
        anchor="center"
    )


bearish_top_table.pack(
    fill="x",
    expand=True
)


bearish_top_table.bind(
    "<Double-1>",
    show_selected_bearish
)


# ==================================================
# K-LINE CHART
# ==================================================

tk.Label(
    content_frame,
    text="K-Line Chart",
    font=("Arial", 15, "bold"),
    bg="#F4F7FB",
    fg="#243B53"
).pack(
    anchor="w",
    padx=25,
    pady=(8, 0)
)


chart_frame = tk.Frame(
    content_frame,
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