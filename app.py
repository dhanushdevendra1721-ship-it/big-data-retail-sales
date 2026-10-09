
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Big Data Retail Sales Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Big Data Analytics for Retail Sales Using Apache Hadoop and Apache Spark")
st.write(
    "A beginner-friendly project to explore retail sales data "
    "and discover useful business insights."
)

st.info(
    "Project setup: We will connect Apache Spark and Hadoop in a later step. "
    "This initial version prepares the sales dashboard."
)

# Sample retail sales data
data = {
    "Date": [
        "2026-01-05", "2026-01-12", "2026-02-03",
        "2026-02-15", "2026-03-04", "2026-03-18",
        "2026-04-06", "2026-04-20", "2026-05-08",
        "2026-05-21"
    ],
    "Product": [
        "Rice", "Sugar", "Cooking Oil", "Rice", "Wheat",
        "Cooking Oil", "Sugar", "Wheat", "Rice", "Cooking Oil"
    ],
    "Category": [
        "Grocery", "Grocery", "Grocery", "Grocery", "Grocery",
        "Grocery", "Grocery", "Grocery", "Grocery", "Grocery"
    ],
    "Quantity": [20, 15, 10, 25, 18, 12, 22, 14, 30, 16],
    "Unit_Price": [60, 45, 150, 60, 40, 150, 45, 40, 60, 150]
}

df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])
df["Revenue"] = df["Quantity"] * df["Unit_Price"]

# Summary metrics
total_revenue = df["Revenue"].sum()
total_units = df["Quantity"].sum()
total_orders = len(df)
top_product = (
    df.groupby("Product")["Quantity"].sum().idxmax()
)

st.subheader("📌 Sales Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
col2.metric("Units Sold", f"{total_units}")
col3.metric("Sales Records", f"{total_orders}")
col4.metric("Top Product", top_product)

# Product filter
st.subheader("🔎 Explore Sales Data")
products = ["All"] + sorted(df["Product"].unique().tolist())
selected_product = st.selectbox("Select a product", products)

if selected_product != "All":
    filtered_df = df[df["Product"] == selected_product]
else:
    filtered_df = df.copy()

st.dataframe(filtered_df, use_container_width=True)

# Product revenue chart
st.subheader("📈 Revenue by Product")
revenue_by_product = (
    filtered_df.groupby("Product")["Revenue"].sum().sort_values()
)

fig, ax = plt.subplots()
revenue_by_product.plot(kind="barh", ax=ax)
ax.set_xlabel("Revenue (₹)")
ax.set_ylabel("Product")
ax.set_title("Retail Revenue by Product")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Monthly revenue chart
st.subheader("📅 Monthly Revenue")
monthly = (
    filtered_df.assign(Month=filtered_df["Date"].dt.to_period("M").astype(str))
    .groupby("Month")["Revenue"]
    .sum()
)

fig2, ax2 = plt.subplots()
monthly.plot(kind="line", marker="o", ax=ax2)
ax2.set_xlabel("Month")
ax2.set_ylabel("Revenue (₹)")
ax2.set_title("Monthly Retail Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.subheader("💡 Business Insights")
st.write(f"- Total revenue in the sample dataset: ₹{total_revenue:,.0f}.")
st.write(f"- Total units sold: {total_units}.")
st.write(f"- Best-selling product by quantity: {top_product}.")
st.write(
    "- This sample dashboard demonstrates basic sales analytics. "
    "Real Hadoop and Spark integration will be configured separately."
)

st.caption(
    "Educational project by Dhanush Devendra | TY BSc Information Technology"
)
