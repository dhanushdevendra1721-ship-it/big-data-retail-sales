
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

st.success(
    "Hadoop HDFS storage and PySpark retail analysis "
    "were demonstrated in Google Colab."
)

st.info(
    "The dashboard below uses a built-in sample dataset and runs "
    "independently. It does not connect directly to the temporary "
    "Google Colab Hadoop or Spark session."
)

# --------------------------------------------------
# SAMPLE RETAIL SALES DATA
# --------------------------------------------------
data = {
    "Date": [
        "2026-01-05", "2026-01-12",
        "2026-02-03", "2026-02-15",
        "2026-03-04", "2026-03-18",
        "2026-04-06", "2026-04-20",
        "2026-05-08", "2026-05-21"
    ],
    "Product": [
        "Rice", "Sugar", "Cooking Oil", "Rice",
        "Wheat", "Cooking Oil", "Sugar", "Wheat",
        "Rice", "Cooking Oil"
    ],
    "Category": ["Grocery"] * 10,
    "Quantity": [20, 15, 10, 25, 18, 12, 22, 14, 30, 16],
    "Unit_Price": [60, 45, 150, 60, 40, 150, 45, 40, 60, 150]
}

df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])
df["Revenue"] = df["Quantity"] * df["Unit_Price"]

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

# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------
filtered_df = df.copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["Date"].dt.date >= start_date)
        & (filtered_df["Date"].dt.date <= end_date)
    ]

products = ["All"] + sorted(filtered_df["Product"].unique().tolist())

selected_product = st.sidebar.selectbox(
    "Select product",
    products
)

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

if not filtered_df.empty:
    top_product = (
        filtered_df.groupby("Product")["Quantity"]
        .sum()
        .idxmax()
    )
else:
    top_product = "N/A"

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
c2.metric("Units Sold", f"{total_units:,}")
c3.metric("Sales Records", f"{total_records}")
c4.metric("Top Product", top_product)

# --------------------------------------------------
# SALES DATA TABLE
# --------------------------------------------------
st.header("🔎 Explore Retail Sales Data")

display_df = filtered_df.copy()
display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

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

    st.dataframe(product_summary, use_container_width=True, hide_index=True)

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
        monthly.groupby("Month")["Revenue"]
        .sum()
        .reset_index()
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
# SPARK-STYLE SUMMARY ANALYTICS
# --------------------------------------------------
st.header("⚡ Big Data Analytics Summary")

if not filtered_df.empty:
    st.write(
        "These summary operations use Pandas in the deployed dashboard. "
        "Equivalent retail analysis was also performed with PySpark in Colab."
    )

    a, b = st.columns(2)

    with a:
        st.subheader("Revenue Statistics")
        st.write(f"**Total revenue:** ₹{total_revenue:,.0f}")
        st.write(f"**Average record revenue:** ₹{filtered_df['Revenue'].mean():,.2f}")
        st.write(f"**Highest record revenue:** ₹{filtered_df['Revenue'].max():,.0f}")
        st.write(f"**Lowest record revenue:** ₹{filtered_df['Revenue'].min():,.0f}")

    with b:
        st.subheader("Quantity Statistics")
        st.write(f"**Total units sold:** {total_units:,}")
        st.write(f"**Average quantity per record:** {filtered_df['Quantity'].mean():,.2f}")
        st.write(f"**Highest quantity in a record:** {filtered_df['Quantity'].max():,}")
        st.write(f"**Products represented:** {filtered_df['Product'].nunique()}")

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

    st.write(f"- Total revenue for the selected data: ₹{total_revenue:,.0f}.")
    st.write(f"- Total quantity sold: {total_units:,} units.")
    st.write(f"- Best-selling product by quantity: {quantity_ranking.index[0]}.")
    st.write(f"- Highest-revenue product: {revenue_ranking.index[0]}.")
    st.write(
        "- Use the sidebar filters to explore different products "
        "and date ranges."
    )
else:
    st.write("Adjust the filters to see business insights.")

# --------------------------------------------------
# PROJECT WORKFLOW
# --------------------------------------------------
st.header("🗄️ Hadoop and Spark Workflow")

st.markdown("""
1. **Retail data preparation:** Created a sample retail sales CSV file.
2. **Hadoop HDFS:** Configured HDFS and stored the CSV file in the `/retail` directory.
3. **Apache Spark:** Read the sales CSV from HDFS using PySpark.
4. **Data analysis:** Calculated total revenue, units sold, and product-wise revenue.
5. **Results export:** Saved the product summary to a CSV file.
6. **Streamlit dashboard:** Displays sales metrics, charts, and business insights using its built-in sample dataset.
""")

st.warning(
    "The Hadoop and Spark steps above describe the workflow demonstrated "
    "in Colab. The deployed dashboard does not currently execute Spark "
    "jobs or read directly from HDFS."
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Developed by Dhanush Devendra | TY BSc Information Technology"
)
st.caption(
    "Technologies: Python • Pandas • Matplotlib • Apache Hadoop • Apache Spark • Streamlit"
)
