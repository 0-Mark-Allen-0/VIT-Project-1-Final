#storage.py
import os
import pandas as pd
import json
import streamlit as st
from typing import Dict, List

from config import CSV_FILE, BUDGET_FILE, SUGGESTIONS_FILE, PRODUCT_ALIASES_FILE, CATEGORIES_FILE, CANON_CATEGORIES

# -----------------------------
# Custom Category Management
# -----------------------------
def load_custom_categories() -> List[str]:
    """Load custom categories from a JSON file."""
    try:
        if os.path.exists(CATEGORIES_FILE):
            with open(CATEGORIES_FILE, 'r') as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError):
        pass
    return []

def save_custom_categories(categories: List[str]):
    """Save custom categories to a JSON file."""
    try:
        with open(CATEGORIES_FILE, 'w') as f:
            json.dump(sorted(list(set(categories))), f, indent=4)
    except IOError:
        st.error("Could not save custom categories.")

def get_all_categories() -> List[str]:
    """Get all categories (canonical + custom), sorted."""
    return sorted(list(set(CANON_CATEGORIES + load_custom_categories())))

def add_custom_category(category: str) -> bool:
    """Add a new custom category if it doesn't already exist."""
    all_categories = get_all_categories()
    if category.title() not in all_categories:
        custom_categories = load_custom_categories()
        custom_categories.append(category.title())
        save_custom_categories(custom_categories)
        return True
    return False

# -----------------------------
# Expense Data & Product Consistency
# -----------------------------
def ensure_csv():
    """Ensure CSV file exists with the proper structure, including the new canonical_product column."""
    columns = ["id", "date", "time", "product", "canonical_product", "category", "amount"]
    try:
        if not os.path.exists(CSV_FILE):
            df = pd.DataFrame(columns=columns)
            df.to_csv(CSV_FILE, index=False)
        else:
            df = pd.read_csv(CSV_FILE)
            # Check for legacy columns and add new ones if they don't exist
            if "id" not in df.columns:
                df.insert(0, "id", range(1, len(df) + 1))
            if "canonical_product" not in df.columns:
                st.info("Upgrading data file for consistency. This is a one-time operation.")
                # Backfill with a normalized version of the product name as a starting point
                df["canonical_product"] = df["product"].str.lower().str.strip()
            
            # Reorder and save to ensure consistent structure going forward
            df = df.reindex(columns=columns, fill_value="")
            df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error ensuring CSV structure: {e}")
        return False

def load_data() -> pd.DataFrame:
    """Load expense data with error handling and data type validation."""
    try:
        if not ensure_csv():
            return pd.DataFrame(columns=["id", "date", "time", "product", "canonical_product", "category", "amount"])
        
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            return df
            
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype(int)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["date"])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(columns=["id", "date", "time", "product", "canonical_product", "category", "amount"])

def load_product_aliases() -> Dict:
    """Load the product alias map from a JSON file."""
    try:
        if os.path.exists(PRODUCT_ALIASES_FILE):
            with open(PRODUCT_ALIASES_FILE, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass # Ignore errors and return an empty dict
    return {}

def save_product_aliases(aliases: Dict):
    """Save the product alias map to a JSON file."""
    try:
        with open(PRODUCT_ALIASES_FILE, 'w') as f:
            json.dump(aliases, f, indent=4)
    except IOError:
        st.error("Could not save product alias map.")

def _resolve_canonical_name(product_name: str, model: str) -> str:
    """
    Resolves the canonical name for a product.
    Uses a cached alias map for performance and falls back to an LLM call for new products.
    """
    # Local import to prevent circular dependency
    from parser import llm_get_canonical
    
    if not product_name:
        return "unknown"

    norm_product = product_name.lower().strip()
    aliases = load_product_aliases()

    # 1. Check cache first for high performance
    if norm_product in aliases:
        return aliases[norm_product]

    # 2. If not in cache, use LLM to determine canonical name
    # Get existing canonical names to guide the LLM
    existing_canonical_names = list(set(aliases.values()))
    
    with st.spinner(f"Normalizing '{product_name}'..."):
        canonical_name = llm_get_canonical(norm_product, existing_canonical_names, model)

    # 3. Update cache and save for future use
    aliases[norm_product] = canonical_name
    save_product_aliases(aliases)

    return canonical_name

def save_row(row: Dict, model: str) -> bool:
    """Save a new expense row, resolving the canonical product name."""
    try:
        df = load_data()

        # --- Resolve canonical name before saving ---
        original_product = row["product"]
        canonical_product = _resolve_canonical_name(original_product, model)
        row["canonical_product"] = canonical_product
        
        new_id = (df["id"].max() + 1) if not df.empty else 1
        row["id"] = new_id
        
        ordered_row = {
            "id": row["id"], 
            "date": row["date"], 
            "time": row["time"], 
            "product": row["product"], 
            "canonical_product": row["canonical_product"],
            "category": row["category"], 
            "amount": row["amount"]
        }
        
        df = pd.concat([df, pd.DataFrame([ordered_row])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        
        update_suggestions(row["canonical_product"], row["category"], row["amount"])
        return True
    except Exception as e:
        st.error(f"Error saving expense: {e}")
        return False
    

def rebuild_suggestions_from_csv():
    """
    Rebuild the entire suggestions database from existing expenses.csv.
    Uses canonical_product for consistency.
    """
    df = load_data()
    
    if df.empty:
        return False
    
    # Ensure canonical_product column exists
    if 'canonical_product' not in df.columns:
        st.warning("canonical_product column missing. Using 'product' instead.")
        product_col = 'product'
    else:
        product_col = 'canonical_product'
    
    # Build suggestions from scratch
    suggestions = {"products": {}}
    
    # Group by canonical product and category, calculate average amount
    grouped = df.groupby([product_col, 'category'])['amount'].agg(['mean', 'count']).reset_index()
    
    for _, row in grouped.iterrows():
        product_key = str(row[product_col]).lower().strip()
        
        if product_key in suggestions["products"]:
            # Product exists with different category - use the one with more entries
            existing = suggestions["products"][product_key]
            if row['count'] > existing['count']:
                suggestions["products"][product_key] = {
                    "category": row['category'],
                    "avg_amount": int(row['mean']),
                    "count": int(row['count'])
                }
        else:
            suggestions["products"][product_key] = {
                "category": row['category'],
                "avg_amount": int(row['mean']),
                "count": int(row['count'])
            }
    
    save_suggestions(suggestions)
    return True

def update_expense(expense_id: int, updated_row: Dict) -> bool:
    """Update an existing expense by its ID."""
    try:
        df = load_data()
        mask = df["id"] == expense_id
        if not mask.any(): return False
            
        for key, value in updated_row.items():
            if key in df.columns:
                df.loc[mask, key] = value
                
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error updating expense: {e}")
        return False

def delete_expense(expense_id: int) -> bool:
    """Delete an expense by its ID."""
    try:
        df = load_data()
        df = df[df["id"] != expense_id]
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error deleting expense: {e}")
        return False

def clear_data() -> bool:
    """Clear all expense, budget, suggestion, and alias data."""
    try:
        # Also clears the new product aliases file and custom categories
        files_to_clear = [CSV_FILE, SUGGESTIONS_FILE, BUDGET_FILE, PRODUCT_ALIASES_FILE, CATEGORIES_FILE]
        for file in files_to_clear:
            if os.path.exists(file):
                os.remove(file)
        ensure_csv()  # Recreate the empty expenses CSV
        return True
    except Exception as e:
        st.error(f"Error clearing data: {e}")
        return False

def upload_csv_data(uploaded_file) -> bool:
    """Upload and validate CSV data, checking against all categories."""
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ["date", "product", "category", "amount"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns: {required_cols}")
            return False
        
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).astype(int)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        
        all_categories = get_all_categories()
        df["category"] = df["category"].apply(lambda x: x if x in all_categories else "Other")
        
        if "canonical_product" not in df.columns:
            df["canonical_product"] = df["product"].str.lower().str.strip()

        existing_df = load_data()
        max_id = existing_df["id"].max() if not existing_df.empty else 0
        df["id"] = range(max_id + 1, max_id + len(df) + 1)
        
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error uploading CSV: {e}")
        return False

def update_product_mapping(original: str, canonical: str):
    """Update product mapping for consistency."""
    aliases = load_product_aliases()
    aliases[original.lower().strip()] = canonical
    save_product_aliases(aliases)

def remap_canonical_product(old_name: str, new_name: str) -> bool:
    """Remaps all instances of an old canonical name to a new one in the CSV."""
    try:
        df = load_data()
        if df.empty:
            return True
        
        df.loc[df['canonical_product'] == old_name, 'canonical_product'] = new_name
        df.to_csv(CSV_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Failed to remap product names: {e}")
        return False

# -----------------------------
# Budget Data (JSON)
# -----------------------------

def load_budgets() -> Dict:
    """Load budget data from JSON file."""
    try:
        if os.path.exists(BUDGET_FILE):
            with open(BUDGET_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_budgets(budgets: Dict) -> bool:
    """Save budget data to JSON file."""
    try:
        with open(BUDGET_FILE, 'w') as f:
            json.dump(budgets, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving budgets: {e}")
        return False

# -----------------------------
# Suggestions Data (JSON)
# -----------------------------

def load_suggestions() -> Dict:
    """Load suggestions data from JSON file."""
    try:
        if os.path.exists(SUGGESTIONS_FILE):
            with open(SUGGESTIONS_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {"products": {}, "categories": {}, "amounts": {}}

def save_suggestions(suggestions: Dict):
    """Save suggestions data to JSON file."""
    try:
        with open(SUGGESTIONS_FILE, 'w') as f:
            json.dump(suggestions, f, indent=4)
    except Exception:
        pass

def update_suggestions(product: str, category: str, amount: int):
    """Update the suggestions database with new expense data."""
    suggestions = load_suggestions()
    
    if "products" not in suggestions:
        suggestions["products"] = {}
    
    product_key = product.lower().strip()
    
    if product_key in suggestions["products"]:
        # Update existing entry
        data = suggestions["products"][product_key]
        old_total = data["avg_amount"] * data["count"]
        new_count = data["count"] + 1
        new_avg = (old_total + amount) / new_count
        
        suggestions["products"][product_key] = {
            "category": category,  # Use most recent category
            "avg_amount": int(new_avg),
            "count": new_count
        }
    else:
        # Create new entry
        suggestions["products"][product_key] = {
            "category": category,
            "avg_amount": amount,
            "count": 1
        }
    
    save_suggestions(suggestions)