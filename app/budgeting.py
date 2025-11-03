#budgeting.py
import datetime as dt
import pandas as pd
from typing import Dict

#UPDATE - Fixing "user created budgets not showing up" bug
from storage import get_all_categories

from config import CANON_CATEGORIES

def get_current_month_key() -> str:
    """Get the current month key in 'YYYY-MM' format."""
    return dt.datetime.now().strftime("%Y-%m")

def calculate_budget_status(df: pd.DataFrame, budgets: Dict) -> Dict:
    """
    Calculate budget vs actual spending for the current month.
    
    Args:
        df (pd.DataFrame): The expenses dataframe.
        budgets (Dict): The budgets dictionary.

    Returns:
        Dict: A dictionary with budget status for each category.
    """
    if df.empty:
        return {}
        
    current_month = get_current_month_key()
    
    # Convert date column to string for filtering if it's not already
    df['date_str'] = df['date'].astype(str)
    current_month_data = df[df["date_str"].str.startswith(current_month)]
    
    if current_month_data.empty:
        return {}
    
    actual_spending = current_month_data.groupby("category")["amount"].sum().to_dict()
    monthly_budgets = budgets.get(current_month, {})
    
    status = {}

    all_categories = get_all_categories()

    for category in all_categories:
        budget = monthly_budgets.get(category, 0)
        actual = actual_spending.get(category, 0)
        
        if budget > 0:
            percentage = (actual / budget) * 100
            status[category] = {
                "budget": budget,
                "actual": actual,
                "remaining": budget - actual,
                "percentage": percentage,
                "status": "over" if actual > budget else "under"
            }
    
    return status
