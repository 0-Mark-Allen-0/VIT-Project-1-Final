#v2
#main.py

# --- START: Load .env file FIRST ---
import os
from dotenv import load_dotenv

# Load environment variables from .env file located in the project root
# Path relative to main.py (inside app_v4) to .env in the parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f".env file loaded from: {dotenv_path}")
else:
    print(f"Warning: .env file not found at {dotenv_path}. Relying on system environment variables or secrets.")
# --- END: Load .env file ---

from typing import Dict, List
import streamlit as st
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go

# Import functions from separated modules
from config import CANON_CATEGORIES, DEFAULT_MODEL
from storage import (
    load_data, save_row, update_expense, delete_expense, clear_data,
    upload_csv_data, load_budgets, save_budgets, get_all_categories,
    add_custom_category, remap_canonical_product, update_product_mapping
)
from budgeting import get_current_month_key, calculate_budget_status
from parser import parse_expense, get_category_suggestions, find_similar_products
from utils import capture_speech, export_data, generate_sample_data

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Expense Tracker",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card-like containers */
    .stContainer {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Better spacing for metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Improved button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# App State Initialization
# -----------------------------
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "edit_id" not in st.session_state:
    st.session_state.edit_id = None
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "pending_category_confirmation" not in st.session_state:
    st.session_state.pending_category_confirmation = None
if "category_suggestions" not in st.session_state:
    st.session_state.category_suggestions = []

# -----------------------------
# Sidebar UI (Redesigned)
# -----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Model Configuration
    with st.expander("Model Configuration", expanded=False):
        model = st.text_input("AI Model", value=DEFAULT_MODEL, help="Gemini model for parsing")
        debug = st.toggle("Debug Mode", value=False, help="Show detailed parsing information")
    
    st.markdown("---")
    
    # Custom Category Management
    with st.expander("📂 Custom Categories", expanded=False):
        st.caption("Add custom categories for better organization")
        custom_category_input = st.text_input(
            "New Category", 
            key="custom_cat_input", 
            placeholder="e.g., Investments"
        )
        if st.button("➕ Add Category", use_container_width=True):
            if custom_category_input and add_custom_category(custom_category_input):
                st.success(f"✅ Added: '{custom_category_input.title()}'")
                st.rerun()
            elif custom_category_input:
                st.warning("Category exists or invalid.")
    
    st.markdown("---")
    
    # Budget Management
    with st.expander("💰 Monthly Budgets", expanded=False):
        st.caption(f"Set budgets for {get_current_month_key()}")
        budgets = load_budgets()
        current_month = get_current_month_key()
        if current_month not in budgets:
            budgets[current_month] = {}
        
        budget_changed = False
        all_categories_for_budget = get_all_categories()
        
        for category in all_categories_for_budget:
            key = f"budget_{category}"
            current_budget = budgets[current_month].get(category, 0)
            new_budget = st.number_input(
                f"{category}", 
                min_value=0, 
                value=current_budget, 
                step=500, 
                key=key,
                help=f"Monthly budget for {category}"
            )
            if new_budget != current_budget:
                budgets[current_month][category] = new_budget
                budget_changed = True
        
        if budget_changed:
            save_budgets(budgets)
            st.success("💾 Budgets updated!")
            st.rerun()
    
    st.markdown("---")
    
    # Data Management
    st.subheader("📊 Data Management")
    df_for_export = load_data()
    
    if not df_for_export.empty:
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox(
                "Format", 
                ["CSV", "JSON", "Excel"],
                label_visibility="collapsed"
            )
        with col2:
            export_bytes = export_data(df_for_export, export_format)
            st.download_button(
                "⬇️ Export", 
                data=export_bytes,
                file_name=f"expenses_{dt.date.today()}.{export_format.lower()}", 
                use_container_width=True
            )
    
    uploaded_file = st.file_uploader("📤 Import CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file:
        if st.button("📥 Import Data", use_container_width=True):
            if upload_csv_data(uploaded_file):
                st.success("✅ Data imported!")
                st.rerun()
    
# Use st.columns(3) to create three equal-width columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Sample", use_container_width=True, help="Load sample data"):
            sample_df = generate_sample_data()
            sample_df.to_csv("expenses.csv", index=False)
            st.success("✅ Loaded!")
            st.rerun()

    with col2:
        if st.button("🔄 Rebuild", use_container_width=True, 
                     help="Rebuild suggestions from expenses.csv"):
            from storage import rebuild_suggestions_from_csv
            with st.spinner("Rebuilding..."):
                if rebuild_suggestions_from_csv():
                    st.success("✅ Suggestions database rebuilt!")
                    st.rerun()
                else:
                    st.error("❌ No data to rebuild from")

    with col3:
        if st.button("🧹 Delete", use_container_width=True, type="secondary", help="Clear all data"):
            if clear_data():
                st.success("✅ Cleared!")
                st.rerun()
# -----------------------------
# Main App UI
# -----------------------------
st.title("💸 Smart Expense Tracker")
st.caption("Your intelligent companion for managing expenses effortlessly")

# Spacer
st.markdown("<br>", unsafe_allow_html=True)

# --- Budget Status Alert (Enhanced) ---
df = load_data()
if not df.empty:
    budgets_for_alert = load_budgets()
    budget_status_alert = calculate_budget_status(df, budgets_for_alert)
    over_budget_cats = [cat for cat, status in budget_status_alert.items() if status["status"] == "over"]
    near_budget_cats = [cat for cat, status in budget_status_alert.items() 
                        if status["status"] == "ok" and status["percentage"] > 80]
    
    if over_budget_cats:
        st.error(f"🚨 **Over Budget:** {', '.join(over_budget_cats)}")
    elif near_budget_cats:
        st.warning(f"⚠️ **Near Limit (>80%):** {', '.join(near_budget_cats)}")

# --- Category Confirmation UI (Enhanced) ---
if st.session_state.pending_category_confirmation:
    st.markdown("### 🤔 Please Confirm Category")
    item = st.session_state.pending_category_confirmation
    suggestions = st.session_state.category_suggestions

    # Display item details in a nice card
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("Product", item['product'])
    with col2:
        st.metric("Amount", f"₹{item['amount']}")
    with col3:
        st.write("")  # Spacer
    
    st.markdown("**Select the most appropriate category:**")
    
    # Create suggestion buttons with confidence indicators
    cols = st.columns(len(suggestions) + 1)
    
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            confidence_emoji = "🟢" if suggestion['confidence'] > 80 else "🟡" if suggestion['confidence'] > 60 else "🔴"
            button_label = f"{confidence_emoji} {suggestion['category']}\n{suggestion['confidence']}%"
            
            if st.button(
                button_label, 
                use_container_width=True, 
                type="primary" if i == 0 else "secondary",
                key=f"sugg_{i}",
                help=suggestion.get('reason', '')
            ):
                item['category'] = suggestion['category']
                now = dt.datetime.now()
                item['date'] = now.date().isoformat()
                item['time'] = now.strftime("%H:%M:%S")
                if save_row(item, model):
                    st.success("✅ Expense saved!")
                    st.session_state.pending_category_confirmation = None
                    st.session_state.category_suggestions = []
                    st.rerun()

    # "Other" button
    with cols[len(suggestions)]:
        if st.button("❓ Other", use_container_width=True):
            item['category'] = "Other"
            now = dt.datetime.now()
            item['date'] = now.date().isoformat()
            item['time'] = now.strftime("%H:%M:%S")
            if save_row(item, model):
                st.success("✅ Saved as 'Other'")
                st.session_state.pending_category_confirmation = None
                st.session_state.category_suggestions = []
                st.rerun()
    
    st.markdown("---")

# --- Main Tabs (Enhanced) ---
tab_add, tab_manage, tab_analytics, tab_budget, tab_consistency = st.tabs(
    ["➕ Add Expense", "✏️ Manage Expenses", "📊 Analytics", "💰 Budget Tracker", "🔄 Data Cleanup"]
)

with tab_add:
    def process_and_save(items: List[Dict], model_name: str):
        """Helper to save items or trigger category confirmation."""
        if len(items) == 1 and items[0]["category"] == "Other":
            item = items[0]
            suggestions = get_category_suggestions(item["product"], model_name)
            if suggestions and suggestions[0]["confidence"] > 65:
                st.session_state.pending_category_confirmation = item
                st.session_state.category_suggestions = suggestions
                st.rerun()
                return

        saved_count = 0
        now = dt.datetime.now()
        for item in items:
            row = {
                "date": now.date().isoformat(),
                "time": now.strftime("%H:%M:%S"),
                "product": item.get("product", "item"),
                "category": item.get("category", "Other"),
                "amount": item.get("amount", 0),
            }
            if save_row(row, model_name):
                saved_count += 1
        
        if saved_count > 0:
            st.success(f"✅ Saved {saved_count} expense(s) successfully!")
            st.rerun()
        else:
            st.error("❌ Could not save expense(s). Please try again.")

    st.markdown("### 📝 Add New Expense")
    st.caption("Use text or voice input to quickly add expenses")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("#### ✍️ Text Input")
        with st.container():
            st.info("💡 **Tips:**\n- Simple: `coffee 50`\n- Multiple: `chips 20, coke 15`\n- Custom category: `laptop as Electronics 40000`")
            
            text_input = st.text_area(
                "Describe your expense", 
                placeholder="e.g., bought coffee for 50 rupees",
                height=100,
                label_visibility="collapsed"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Parse & Save", use_container_width=True, type="primary"):
                if not text_input.strip():
                    st.warning("⚠️ Please enter an expense description.")
                else:
                    with st.spinner("🔍 Analyzing your expense..."):
                        success, items, error_msg = parse_expense(text_input, model, debug=debug)
                    
                    if success:
                        process_and_save(items, model)
                    else:
                        st.error(f"❌ {error_msg}")

    with col2:
        st.markdown("#### 🎙️ Voice Input")
        with st.container():
            st.info("💡 **How to use:**\n1. Click 'Start Recording'\n2. Speak your expense\n3. Click 'Parse & Save'")
            
            if st.button("🎤 Start Recording", use_container_width=True, type="primary"):
                with st.spinner("🎧 Listening..."):
                    transcript = capture_speech()
                    if transcript:
                        st.session_state["last_transcript"] = transcript
                        st.success("✅ Recording captured!")

            transcript = st.session_state.get("last_transcript", "")
            st.text_area(
                "Transcript", 
                value=transcript, 
                height=100, 
                disabled=True,
                label_visibility="collapsed",
                placeholder="Your voice input will appear here..."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button(
                "🚀 Parse & Save", 
                use_container_width=True, 
                disabled=not bool(transcript.strip()),
                key="voice_parse"
            ):
                with st.spinner("🔍 Analyzing your expense..."):
                    success, items, error_msg = parse_expense(transcript, model, debug=debug)
                
                if success:
                    st.session_state["last_transcript"] = ""
                    process_and_save(items, model)
                else:
                    st.error(f"❌ {error_msg}")

with tab_manage:
    st.markdown("### ✏️ Manage Your Expenses")
    st.caption("Search, filter, edit, and delete your expenses")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if df.empty:
        st.info("📭 No expenses yet. Start adding some from the 'Add Expense' tab!")
    else:
        # Filters in a cleaner layout
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                search_term = st.text_input(
                    "🔍 Search", 
                    placeholder="Search products...",
                    label_visibility="collapsed"
                )
            with col2:
                all_categories_for_manage = get_all_categories()
                category_filter = st.selectbox(
                    "Category", 
                    ["All"] + all_categories_for_manage,
                    label_visibility="collapsed"
                )
            with col3:
                date_filter = st.date_input(
                    "Date", 
                    value=None,
                    label_visibility="collapsed"
                )
            with col4:
                st.write("")  # Spacer for alignment
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Apply filters
        filtered_df = df.copy()
        if search_term:
            mask = filtered_df["product"].str.contains(search_term, case=False, na=False) | \
                   filtered_df["canonical_product"].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df["category"] == category_filter]
        if date_filter:
            filtered_df = filtered_df[filtered_df["date"] == date_filter]
        
        if filtered_df.empty:
            st.info("🔍 No expenses match your filters. Try adjusting your search criteria.")
        else:
            # Results counter
            st.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** expenses")
            
            # Display expenses in cards
            for _, row in filtered_df.iterrows():
                expense_id = row["id"]
                is_editing = st.session_state.edit_mode and st.session_state.edit_id == expense_id
                
                with st.container():
                    if is_editing:
                        col1, col2, col3, col4, col5, col6 = st.columns([2.5, 2, 1.5, 1.5, 0.8, 0.8])
                        
                        new_product = col1.text_input(
                            "Product", 
                            value=row["product"], 
                            key=f"edit_product_{expense_id}",
                            label_visibility="collapsed"
                        )
                        new_category = col2.selectbox(
                            "Category", 
                            get_all_categories(), 
                            index=get_all_categories().index(row["category"]), 
                            key=f"edit_category_{expense_id}",
                            label_visibility="collapsed"
                        )
                        new_amount = col3.number_input(
                            "Amount", 
                            value=int(row["amount"]), 
                            min_value=0, 
                            key=f"edit_amount_{expense_id}",
                            label_visibility="collapsed"
                        )
                        col4.write(f"📅 {row['date']}")
                        
                        if col5.button("💾", help="Save changes", key=f"save_{expense_id}", use_container_width=True):
                            updated_row = {
                                "product": new_product, 
                                "category": new_category, 
                                "amount": new_amount
                            }
                            if update_expense(expense_id, updated_row):
                                st.success("✅ Updated!")
                                st.session_state.edit_mode = False
                                st.session_state.edit_id = None
                                st.rerun()
                        
                        if col6.button("❌", help="Cancel", key=f"cancel_{expense_id}", use_container_width=True):
                            st.session_state.edit_mode = False
                            st.session_state.edit_id = None
                            st.rerun()
                    else:
                        col1, col2, col3, col4, col5, col6 = st.columns([2.5, 2, 1.5, 1.5, 0.8, 0.8])
                        
                        col1.markdown(f"**{row['product']}**")
                        col2.markdown(f"`{row['category']}`")
                        col3.markdown(f"**₹{row['amount']:,}**")
                        col4.write(f"📅 {row['date']}")
                        
                        if col5.button("✏️", help="Edit", key=f"edit_{expense_id}", use_container_width=True):
                            st.session_state.edit_mode = True
                            st.session_state.edit_id = expense_id
                            st.rerun()
                        
                        if col6.button("🗑️", help="Delete", key=f"delete_{expense_id}", use_container_width=True):
                            if delete_expense(expense_id):
                                st.success("✅ Deleted!")
                                st.rerun()
                    
                    st.markdown("<hr style='margin: 0.5rem 0; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

with tab_analytics:
    st.markdown("### 📊 Analytics Dashboard")
    st.caption("Visualize and understand your spending patterns")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if df.empty:
        st.info("📊 No data to analyze yet. Add some expenses or load sample data to see insights!")
    else:
        if "canonical_product" not in df.columns:
            st.error("⚠️ Data format issue detected. Please clear and reload your data.")
        else:
            # Key Metrics Section
            st.markdown("#### 📈 Quick Stats")
            col1, col2, col3, col4 = st.columns(4)
            
            total_expenses = df["amount"].sum()
            avg_expense = df["amount"].mean()
            expense_count = len(df)
            top_category = df.groupby("category")["amount"].sum().idxmax()
            
            with col1:
                st.metric(
                    "Total Spent", 
                    f"₹{total_expenses:,}",
                    delta=None,
                    help="Total amount spent across all expenses"
                )
            with col2:
                st.metric(
                    "Avg Expense", 
                    f"₹{avg_expense:.0f}",
                    help="Average amount per expense"
                )
            with col3:
                st.metric(
                    "Total Expenses", 
                    f"{expense_count}",
                    help="Total number of expense entries"
                )
            with col4:
                st.metric(
                    "Top Category", 
                    top_category,
                    help="Category with highest spending"
                )
            
            st.markdown("---")
            
            # Charts Section 1
            chart_col1, chart_col2 = st.columns(2, gap="large")
            
            with chart_col1:
                st.markdown("#### 🥧 Spending by Category")
                cat_sum = df.groupby("category")["amount"].sum().reset_index()
                cat_sum = cat_sum.sort_values("amount", ascending=False)
                
                fig_pie = px.pie(
                    cat_sum, 
                    values="amount", 
                    names="category",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with chart_col2:
                st.markdown("#### 📈 Spending Trend")
                
                # Dynamic controls
                trend_df = df.copy()
                trend_df['date'] = pd.to_datetime(trend_df['date'])
                available_months = sorted(trend_df['date'].dt.strftime('%Y-%m').unique(), reverse=True)
                
                control_col1, control_col2 = st.columns(2)
                with control_col1:
                    selected_month = st.selectbox(
                        "Period",
                        options=["All Time"] + available_months,
                        key="trend_month_select"
                    )
                with control_col2:
                    agg_period = st.radio(
                        "View",
                        options=["Daily", "Weekly"],
                        key="trend_agg_period",
                        horizontal=True
                    )
                
                # Filter data
                if selected_month != "All Time":
                    trend_df = trend_df[trend_df['date'].dt.strftime('%Y-%m') == selected_month]
                
                # Aggregate data
                if agg_period == "Daily":
                    spending_trend = trend_df.groupby('date')['amount'].sum().reset_index()
                    x_axis = 'date'
                else:
                    spending_trend = trend_df.set_index('date')['amount'].resample('W-MON').sum().reset_index()
                    x_axis = 'date'
                
                if spending_trend.empty:
                    st.info("No data for selected period")
                else:
                    fig_line = px.line(
                        spending_trend, 
                        x=x_axis, 
                        y="amount",
                        markers=True,
                        line_shape='spline'
                    )
                    fig_line.update_traces(line_color='#FF6B6B', marker=dict(size=8))
                    fig_line.update_xaxes(showticklabels=False, title_text=f"{agg_period} ({selected_month})")
                    fig_line.update_yaxes(tickprefix="₹", title_text="Amount")
                    fig_line.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_line, use_container_width=True)
            
            st.markdown("---")
            
            # Charts Section 2
            chart_col3, chart_col4 = st.columns(2, gap="large")
            
            with chart_col3:
                st.markdown("#### 🏆 Top 10 Products")
                top_products = df.groupby("canonical_product")["amount"].sum().nlargest(10)
                
                fig_bar = px.bar(
                    x=top_products.values, 
                    y=top_products.index, 
                    orientation='h',
                    color=top_products.values,
                    color_continuous_scale='Blues'
                )
                fig_bar.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    xaxis_title="Total Spent (₹)",
                    yaxis_title="",
                    showlegend=False,
                    height=400,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with chart_col4:
                st.markdown("#### 📅 Monthly Comparison")
                df_monthly = df.copy()
                df_monthly['date'] = pd.to_datetime(df_monthly['date'])
                df_monthly['month'] = df_monthly['date'].dt.strftime('%Y-%m')
                
                monthly_spend = df_monthly.groupby("month")["amount"].sum().reset_index()
                monthly_spend = monthly_spend.sort_values("month")
                
                fig_monthly = px.bar(
                    monthly_spend, 
                    x="month", 
                    y="amount",
                    color="amount",
                    color_continuous_scale='Greens'
                )
                fig_monthly.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Total Spent (₹)",
                    showlegend=False,
                    height=400,
                    coloraxis_showscale=False
                )
                fig_monthly.update_yaxes(tickprefix="₹")
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            st.markdown("---")
            
            # Recent Expenses Table
            st.markdown("#### 🕐 Recent Transactions")
            recent_df = df.sort_values(["date", "time"], ascending=False).head(15)
            st.dataframe(
                recent_df[["date", "product", "canonical_product", "category", "amount"]], 
                use_container_width=True,
                column_config={
                    "date": "Date",
                    "product": "Original Input",
                    "canonical_product": "Cleaned Name",
                    "category": "Category",
                    "amount": st.column_config.NumberColumn(
                        "Amount",
                        format="₹%d"
                    )
                },
                hide_index=True
            )

with tab_budget:
    st.markdown("### 💰 Budget Tracker")
    st.caption(f"Monitor your spending against budgets for {get_current_month_key()}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    budgets_for_tab = load_budgets()
    current_month = get_current_month_key()
    
    if df.empty:
        st.info("📊 No expenses yet. Add some expenses to see budget analysis.")
    else:
        budget_status = calculate_budget_status(df, budgets_for_tab)
        
        if not budget_status:
            st.info("💡 Set budgets in the sidebar to track your spending limits!")
        else:
            # Budget overview cards
            for category, status in budget_status.items():
                with st.container():
                    # Status indicator
                    if status["status"] == "over":
                        status_emoji = "🔴"
                        status_color = "#ff4444"
                    elif status['percentage'] > 80:
                        status_emoji = "🟡"
                        status_color = "#ffaa00"
                    else:
                        status_emoji = "🟢"
                        status_color = "#00cc44"
                    
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                    
                    with col1:
                        st.markdown(f"### {status_emoji} **{category}**")
                    with col2:
                        st.metric("Budget", f"₹{status['budget']:,}")
                    with col3:
                        st.metric("Spent", f"₹{status['actual']:,}")
                    with col4:
                        remaining_color = "inverse" if status['remaining'] < 0 else "off"
                        st.metric(
                            "Remaining", 
                            f"₹{abs(status['remaining']):,}",
                            delta=None
                        )
                    
                    # Progress bar
                    progress = min(status['percentage'] / 100, 1.0)
                    st.progress(progress)
                    
                    # Status message
                    if status['percentage'] > 100:
                        st.error(f"⚠️ Over budget by ₹{status['actual'] - status['budget']:,} ({status['percentage']:.1f}%)")
                    elif status['percentage'] > 80:
                        st.warning(f"⚠️ You've used {status['percentage']:.1f}% of your budget")
                    else:
                        st.success(f"✅ {status['percentage']:.1f}% used - {100 - status['percentage']:.1f}% remaining")
                    
                    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visual comparison chart
            st.markdown("#### 📊 Budget vs Actual Comparison")
            budget_data = [{"Category": cat, **stat} for cat, stat in budget_status.items()]
            budget_df = pd.DataFrame(budget_data)
            
            if not budget_df.empty:
                budget_melted = budget_df.melt(
                    id_vars=["Category"], 
                    value_vars=["budget", "actual"],
                    var_name="Type", 
                    value_name="Amount"
                )
                
                fig_budget = px.bar(
                    budget_melted, 
                    x="Category", 
                    y="Amount", 
                    color="Type",
                    barmode="group",
                    color_discrete_map={
                        "budget": "#4CAF50",
                        "actual": "#2196F3"
                    }
                )
                fig_budget.update_layout(
                    xaxis_title="Category",
                    yaxis_title="Amount (₹)",
                    legend_title="",
                    height=450
                )
                fig_budget.update_yaxes(tickprefix="₹")
                st.plotly_chart(fig_budget, use_container_width=True)

with tab_consistency:
    st.markdown("### 🔄 Data Consistency Manager")
    st.caption("Keep your expense data clean by merging duplicate or similar product names")
    
    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty or 'canonical_product' not in df.columns:
        st.info("📭 No data available for consistency analysis. Add some expenses first!")
    else:
        canonical_products = sorted(df['canonical_product'].unique())
        potential_merges = {}

        # Find similar products
        with st.spinner("🔍 Analyzing product names..."):
            for i, prod1 in enumerate(canonical_products):
                similar = find_similar_products(prod1, canonical_products[i+1:], threshold=0.75)
                if similar:
                    potential_merges[prod1] = [s[0] for s in similar]

        if not potential_merges:
            st.success("✅ All product names look consistent! No duplicates found.")
        else:
            st.warning(f"🔍 Found **{len(potential_merges)}** potential duplicate sets that can be merged")
            
            st.markdown("---")
            
            # Display merge suggestions
            for target_product, similar_list in potential_merges.items():
                with st.expander(f"🔀 Merge into **{target_product}**", expanded=True):
                    st.markdown(f"**Target:** `{target_product}`")
                    st.markdown("**Similar items found:**")
                    
                    # Create a grid for better layout
                    for i in range(0, len(similar_list), 2):
                        cols = st.columns([4, 1, 4, 1])
                        
                        # First item in row
                        similar_product = similar_list[i]
                        count = len(df[df['canonical_product'] == similar_product])
                        cols[0].write(f"• `{similar_product}` ({count} entries)")
                        
                        if cols[1].button(
                            "✓ Merge", 
                            key=f"merge_{similar_product}_into_{target_product}",
                            use_container_width=True,
                            type="primary"
                        ):
                            with st.spinner(f"Merging '{similar_product}' into '{target_product}'..."):
                                update_product_mapping(similar_product, target_product)
                                remap_canonical_product(similar_product, target_product)
                            st.success(f"✅ Merged successfully!")
                            st.rerun()
                        
                        # Second item in row (if exists)
                        if i + 1 < len(similar_list):
                            similar_product2 = similar_list[i + 1]
                            count2 = len(df[df['canonical_product'] == similar_product2])
                            cols[2].write(f"• `{similar_product2}` ({count2} entries)")
                            
                            if cols[3].button(
                                "✓ Merge", 
                                key=f"merge_{similar_product2}_into_{target_product}",
                                use_container_width=True,
                                type="primary"
                            ):
                                with st.spinner(f"Merging '{similar_product2}' into '{target_product}'..."):
                                    update_product_mapping(similar_product2, target_product)
                                    remap_canonical_product(similar_product2, target_product)
                                st.success(f"✅ Merged successfully!")
                                st.rerun()
                    
                    st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem 0;'>
        <p>💸 Smart Expense Tracker | VIT Project 1</p>
    </div>
    """,
    unsafe_allow_html=True
)