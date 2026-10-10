
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Big Data Retail Sales Analytics",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# PROJECT HEADER
# --------------------------------------------------
st.title("📊 Big Data Analytics for Retail Sales")
st.subheader("Using Apache Hadoop and Apache Spark")

st.write(
    "An educational project demonstrating retail sales analytics, "
    "Hadoop Distributed File System (HDFS) storage, and PySpark analysis."
)

# --------------------------------------------------
# LOAD SPARK-PROCESSED DATA
# --------------------------------------------------
SPARK_FILE = Path(__file__).parent / "spark_retail_sales_results.csv"
RAW_FILE = Path(__file__).parent / "retail_sales.csv"

@st.cache_data
def load_sales_data():
    if SPARK_FILE.exists():
        data = pd.read_csv(SPARK_FILE)
        source = "Spark-generated results CSV"
    elif RAW_FILE.exists():
        data = pd.read_csv(RAW_FILE)
        source = "Raw retail sales CSV (fallback)"
    else:
        return pd.DataFrame(), "No data file found"

    required_columns = [
        "Date", "Product", "Category",
        "Quantity", "Unit_Price"
    ]

    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing required CSV columns: " + ", ".join(missing)
        )

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Unit_Price"] = pd.to_numeric(data["Unit_Price"], errors="coerce")
    data = data.dropna(
        subset=["Date", "Product", "Quantity", "Unit_Price"]
    ).copy()

    # Use Spark's Revenue column when present; otherwise calculate it.
    if "Revenue" in data.columns:
        data["Revenue"] = pd.to_numeric(data["Revenue"], errors="coerce")
        data["Revenue"] = data["Revenue"].fillna(
            data["Quantity"] * data["Unit_Price"]
        )
    else:
        data["Revenue"] = data["Quantity"] * data["Unit_Price"]

    return data, source

try:
    df, data_source = load_sales_data()
except Exception as error:
    st.error(f"Unable to load sales data: {error}")
    st.stop()

if df.empty:
    st.error(
        "No sales data was found. Please add retail_sales.csv "
        "or spark_retail_sales_results.csv to the project repository."
    )
    st.stop()

st.success(f"Sales data loaded from: {data_source}")

if SPARK_FILE.exists():
    st.info(
        "The dashboard is displaying results exported from a Spark "
        "processing run in Google Colab. Spark is not executing live "
        "inside this Streamlit deployment."
    )
else:
    st.warning(
        "The Spark results file was not found. The dashboard is using "
        "the raw CSV fallback; these records have not been processed "
        "by Spark at dashboard runtime."
    )

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Dashboard Controls")
st.sidebar.write("Project: Retail Sales Analytics")
st.sidebar.write("Technologies: Python, Pandas, PySpark, Hadoop")

date_min = df["Date"].min().date()
date_max = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

filtered_df = df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["Date"].dt.date >= start_date)
        & (filtered_df["Date"].dt.date <= end_date)
    ]

products = ["All"] + sorted(filtered_df["Product"].unique().tolist())

selected_product = st.sidebar.selectbox("Select product", products)

if selected_product != "All":
    filtered_df = filtered_df[
        filtered_df["Product"] == selected_product
    ]

# --------------------------------------------------
# KEY PERFORMANCE INDICATORS
# --------------------------------------------------
st.header("📌 Sales Overview")

total_revenue = filtered_df["Revenue"].sum()
total_units = filtered_df["Quantity"].sum()
total_records = len(filtered_df)

top_product = (
    filtered_df.groupby("Product")["Quantity"].sum().idxmax()
    if not filtered_df.empty else "N/A"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
c2.metric("Units Sold", f"{total_units:,.0f}")
c3.metric("Sales Records", f"{total_records}")
c4.metric("Top Product", top_product)

# --------------------------------------------------
# SALES DATA TABLE
# --------------------------------------------------
st.header("🔎 Explore Retail Sales Data")

display_df = filtered_df.copy()
display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# --------------------------------------------------
# PRODUCT-WISE ANALYTICS
# --------------------------------------------------
st.header("📦 Product-Wise Analysis")

if not filtered_df.empty:
    product_summary = (
        filtered_df.groupby("Product")
        .agg(
            Total_Quantity=("Quantity", "sum"),
            Total_Revenue=("Revenue", "sum"),
            Average_Unit_Price=("Unit_Price", "mean")
        )
        .sort_values("Total_Revenue", ascending=False)
        .reset_index()
    )

    st.dataframe(
        product_summary,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    chart_data = product_summary.sort_values("Total_Revenue")
    ax.barh(chart_data["Product"], chart_data["Total_Revenue"])
    ax.set_xlabel("Revenue (₹)")
    ax.set_ylabel("Product")
    ax.set_title("Revenue by Product")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("No sales records match the selected filters.")

# --------------------------------------------------
# MONTHLY REVENUE
# --------------------------------------------------
st.header("📅 Monthly Revenue")

if not filtered_df.empty:
    monthly = filtered_df.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)

    monthly_summary = (
        monthly.groupby("Month")["Revenue"].sum().reset_index()
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(
        monthly_summary["Month"],
        monthly_summary["Revenue"],
        marker="o"
    )
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Revenue (₹)")
    ax2.set_title("Monthly Retail Revenue")
    ax2.tick_params(axis="x", rotation=45)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# --------------------------------------------------
# REVENUE AND QUANTITY STATISTICS
# --------------------------------------------------
st.header("⚡ Big Data Analytics Summary")

if not filtered_df.empty:
    a, b = st.columns(2)

    with a:
        st.subheader("Revenue Statistics")
        st.write(f"**Total revenue:** ₹{total_revenue:,.0f}")
        st.write(
            f"**Average record revenue:** "
            f"₹{filtered_df['Revenue'].mean():,.2f}"
        )
        st.write(
            f"**Highest record revenue:** "
            f"₹{filtered_df['Revenue'].max():,.0f}"
        )
        st.write(
            f"**Lowest record revenue:** "
            f"₹{filtered_df['Revenue'].min():,.0f}"
        )

    with b:
        st.subheader("Quantity Statistics")
        st.write(f"**Total units sold:** {total_units:,.0f}")
        st.write(
            f"**Average quantity per record:** "
            f"{filtered_df['Quantity'].mean():,.2f}"
        )
        st.write(
            f"**Highest quantity in a record:** "
            f"{filtered_df['Quantity'].max():,.0f}"
        )
        st.write(
            f"**Products represented:** "
            f"{filtered_df['Product'].nunique()}"
        )

# --------------------------------------------------
# BUSINESS INSIGHTS
# --------------------------------------------------
st.header("💡 Business Insights")

if not filtered_df.empty:
    revenue_ranking = (
        filtered_df.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    quantity_ranking = (
        filtered_df.groupby("Product")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    st.write(f"- Total revenue for selected data: ₹{total_revenue:,.0f}.")
    st.write(f"- Total quantity sold: {total_units:,.0f} units.")
    st.write(
        f"- Best-selling product by quantity: {quantity_ranking.index[0]}."
    )
    st.write(
        f"- Highest-revenue product: {revenue_ranking.index[0]}."
    )
    st.write("- Use the sidebar filters to explore the sales data.")
else:
    st.write("Adjust the filters to see business insights.")

# --------------------------------------------------
# PROJECT WORKFLOW
# --------------------------------------------------
st.header("🗄️ Hadoop and Spark Workflow")

st.markdown("""
1. **Retail data preparation:** Prepared retail sales data as a CSV file.
2. **Hadoop HDFS:** HDFS storage was demonstrated separately in Google Colab.
3. **Apache Spark:** PySpark read and processed the sales CSV in Colab.
4. **Data analysis:** Spark calculated revenue and quantity totals.
5. **Results export:** Spark output was exported as a CSV file.
6. **Streamlit dashboard:** Displays the exported Spark results and provides
   interactive filters, charts, and business insights.
""")

st.warning(
    "The deployed dashboard reads a previously exported Spark results CSV. "
    "It does not start Spark jobs or connect directly to HDFS at runtime."
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Developed by Dhanush Devendra | TY BSc Information Technology"
)
st.caption(
    "Technologies: Python • Pandas • Matplotlib • Apache Hadoop • "
    "Apache Spark • Streamlit"
)
