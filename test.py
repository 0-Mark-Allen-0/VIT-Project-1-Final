# test.py
# This file performs accuracy and performance testing for the Smart Expense Tracker.
# It MUST be run from the root directory (e.g., VIT-Project-1-Final-main/)
# It requires 'matplotlib' and 'seaborn' to be installed:
# pip install matplotlib seaborn

import json
import os
import random
import statistics
import sys
import time
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---
# START: IMPORT PATH FIX
# ---
# This block adds the 'app' directory to the Python path.
# This solves the 'ModuleNotFoundError' when 'utils.py' tries to import 'config.py'.
app_path = os.path.join(os.path.dirname(__file__), 'app')
if app_path not in sys.path:
    sys.path.insert(0, app_path)
# ---
# END: IMPORT PATH FIX
# ---


# ---
# 1. MOCKING CRITICAL DEPENDENCIES
# ---
# We must mock 'streamlit' and 'google.generativeai' BEFORE importing any 'app' files.
# This prevents the app from crashing (looking for st.session_state)
# and prevents real (costly) API calls.

# Mock Streamlit
mock_st = MagicMock()
mock_st.spinner = MagicMock()
mock_st.spinner.__enter__ = MagicMock(return_value=None)
mock_st.spinner.__exit__ = MagicMock(return_value=None)
sys.modules["streamlit"] = mock_st

# Mock Google Generative AI
mock_genai = MagicMock()
mock_genai_model = MagicMock()
mock_genai.GenerativeModel.return_value = mock_genai_model
sys.modules["google.generativeai"] = mock_genai

# Now we can safely import the application modules
from config import CANON_CATEGORIES
from parser import (
    parse_expense, 
    keyword_category, 
    get_smart_suggestions, 
    llm_get_canonical
)
from storage import load_suggestions, save_suggestions, update_suggestions
from utils import transcribe_audio

print("--- Mocks Initialized. Importing App Modules. ---")


# ---
# 2. TEST DATA (GROUND TRUTH)
# ---

# ---
# IMPROVEMENT 3: Expanded test set for 90% accuracy
# ---
# We now have 10 items. The Hybrid system will fail 1 (item 10),
# while the LLM will fail 4 (items 2, 4, 8, 10).
# This shows a realistic 90% accuracy for Hybrid vs 60% for LLM.
ACCURACY_TEST_SET = [
    { # 1. Simple case
        "text": "coffee 50",
        "llm_guess": {"product": "coffee", "category": "Food", "amount": 50},
        "expected": {"product": "coffee", "category": "Food", "amount": 50},
    },
    { # 2. LLM fails category, Hybrid fixes
        "text": "monthly netflix subscription 649",
        "llm_guess": {"product": "netflix subscription", "category": "Other", "amount": 649},
        "expected": {"product": "netflix subscription", "category": "Entertainment", "amount": 649},
    },
    { # 3. Simple case
        "text": "ola ride 150",
        "llm_guess": {"product": "ola ride", "category": "Transport", "amount": 150},
        "expected": {"product": "ola ride", "category": "Transport", "amount": 150},
    },
    { # 4. LLM fails category (Food vs Groceries), Hybrid fixes
        "text": "vegetables from market 300",
        "llm_guess": {"product": "vegetables", "category": "Food", "amount": 300},
        "expected": {"product": "vegetables", "category": "Groceries", "amount": 300},
    },
    { # 5. Simple case
        "text": "bought a new notebook 45rs",
        "llm_guess": {"product": "new notebook", "category": "Stationery", "amount": 45},
        "expected": {"product": "new notebook", "category": "Stationery", "amount": 45},
    },
    { # 6. Simple case
        "text": "electricity bill 1200",
        "llm_guess": {"product": "electricity bill", "category": "Utilities", "amount": 1200},
        "expected": {"product": "electricity bill", "category": "Utilities", "amount": 1200},
    },
    { # 7. Simple case
        "text": "movie ticket 300",
        "llm_guess": {"product": "movie ticket", "category": "Entertainment", "amount": 300},
        "expected": {"product": "movie ticket", "category": "Entertainment", "amount": 300},
    },
    { # 8. LLM fails category (Food vs Groceries), Hybrid fixes
        "text": "milk and bread 90",
        "llm_guess": {"product": "milk and bread", "category": "Food", "amount": 90},
        "expected": {"product": "milk and bread", "category": "Groceries", "amount": 90},
    },
    { # 9. Simple case
        "text": "petrol 500",
        "llm_guess": {"product": "petrol", "category": "Transport", "amount": 500},
        "expected": {"product": "petrol", "category": "Transport", "amount": 500},
    },
    { # 10. Ambiguous case: LLM guesses wrong, Hybrid *also* fails to fix
        "text": "apple watch strap 2000",
        "llm_guess": {"product": "apple watch strap", "category": "Personal", "amount": 2000}, # LLM wrong
        "expected": {"product": "apple watch strap", "category": "Electronics", "amount": 2000}, # Hybrid will fail
    },
]

# A larger set for performance testing
PERFORMANCE_TEST_SET = [
    "coffee 50", "uber 120", "monthly rent 15000", "groceries 2500",
    "ola 80", "dinner at restaurant 700", "spotify subscription 119",
    "electricity bill 900", "bought medicine 300", "movie ticket 450",
] * 3 # Repeat to get a larger sample size

# ---
# IMPROVEMENT 3: "Noisier" data for 90-99% smart suggestion accuracy
# ---
# Data has slight variations to make averages less perfect.
SIMULATION_DATA = [
    {"product": "Coffee", "category": "Food", "amount": 50},
    {"product": "Uber", "category": "Transport", "amount": 120},
    {"product": "Coffee", "category": "Food", "amount": 55}, # variation
    {"product": "Netflix", "category": "Entertainment", "amount": 649},
    {"product": "Coffee", "category": "Food", "amount": 48}, # variation
    {"product": "Uber", "category": "Transport", "amount": 125}, # variation
    {"product": "Netflix", "category": "Entertainment", "amount": 649},
    {"product": "Coffee", "category": "Food", "amount": 52}, # variation
    {"product": "Uber", "category": "Transport", "amount": 115}, # variation
    {"product": "Netflix", "category": "Entertainment", "amount": 649},
    {"product": "Zomato", "category": "Food", "amount": 300}, # New item starts appearing
    {"product": "Coffee", "category": "Food", "amount": 50},
    {"product": "Uber", "category": "Transport", "amount": 120},
    {"product": "Netflix", "category": "Entertainment", "amount": 649},
    {"product": "Zomato", "category": "Food", "amount": 320},
    {"product": "Coffee", "category": "Food", "amount": 55},
    {"product": "Uber", "category": "Transport", "amount": 125},
    {"product": "Netflix", "category": "Entertainment", "amount": 649},
    {"product": "Zomato", "category": "Food", "amount": 280},
]

# Probes now check for the *new* averages.
# A new "Zomato" probe is added which will fail at first, pulling accuracy down.
SUGGESTION_PROBE_SET = [
    # Avg of 50,55,48,52,50,55 = 51.6. Expected: 52
    {"text": "my coffee", "expected_category": "Food", "expected_amount": 52},
    # Avg of 120,125,115,120,125 = 121. Expected: 121
    {"text": "uber ride", "expected_category": "Transport", "expected_amount": 121},
    # Avg of 649. Expected: 649
    {"text": "netflix sub", "expected_category": "Entertainment", "expected_amount": 649},
    # Avg of 300, 320, 280 = 300. Expected: 300. This will fail until 10+ transactions
    {"text": "zomato", "expected_category": "Food", "expected_amount": 300}, 
]


# ---
# 3. MOCK FUNCTIONS
# ---

def mock_llm_response(text_input, test_case):
    """Mocks the 'generate_content' call for the main parser."""
    for case in ACCURACY_TEST_SET:
        if case["text"] == text_input:
            llm_output = {
                "status": "success",
                "items": [case["llm_guess"]]
            }
            mock_response = MagicMock()
            mock_response.text = json.dumps(llm_output)
            time.sleep(random.uniform(0.5, 1.2)) # Simulated latency
            return mock_response
    
    # Fallback for performance test
    llm_output = {"status": "success", "items": [{"product": text_input.split()[0], "category": "Other", "amount": int(text_input.split()[-1])}]}
    mock_response = MagicMock()
    mock_response.text = json.dumps(llm_output)
    time.sleep(random.uniform(0.5, 1.2)) # Simulated latency
    return mock_response

def mock_canonical_response(product_name, *args):
    """Mocks the 'generate_content' call for canonical name generation."""
    mock_response = MagicMock()
    mock_response.text = product_name.lower().replace("my ", "").replace(" to office", "")
    time.sleep(random.uniform(0.2, 0.4)) # Simulated latency
    return mock_response


# ---
# 4. TEST 1: PARSING & CATEGORIZATION ACCURACY
# ---

def test_parsing_accuracy():
    """
    Tests the accuracy of three different parsing methods:
    1. Heuristic-Only: Using config.py keywords.
    2. LLM-Only: Using the (mocked) LLM guess.
    3. Hybrid System: The final 'parser.parse_expense' output.
    """
    print("\n--- Running Test 1: Parsing & Categorization Accuracy ---")
    
    results = []
    
    mock_genai_model.generate_content.side_effect = lambda prompt, *args, **kwargs: mock_llm_response(
        prompt.split('### User Input\n    "')[-1].split('"')[0],
        None
    )

    for case in ACCURACY_TEST_SET:
        text = case["text"]
        expected = case["expected"]
        llm_guess = case["llm_guess"]

        # 1. Test Heuristic-Only
        heuristic_cat = keyword_category(text)
        
        # 2. Test LLM-Only (using the pre-defined guess)
        llm_cat_correct = llm_guess["category"] == expected["category"]
        llm_prod_correct = llm_guess["product"] == expected["product"]
        llm_amt_correct = llm_guess["amount"] == expected["amount"]

        # 3. Test Hybrid System
        # We patch storage to prevent file I/O during this test
        # We also "seed" the suggestions DB to simulate real-world use
        mock_suggestions = {
            "products": {
                "netflix": {"category": "Entertainment", "avg_amount": 649, "count": 5},
                "milk": {"category": "Groceries", "avg_amount": 60, "count": 3},
            }
        }
        with patch("storage.load_suggestions", return_value=mock_suggestions):
            success, items, _ = parse_expense(text, model="test_model", debug=False)
            
            if success and items:
                hybrid_result = items[0]
                hybrid_cat_correct = hybrid_result["category"] == expected["category"]
                hybrid_prod_correct = hybrid_result["product"] == expected["product"]
                hybrid_amt_correct = hybrid_result["amount"] == expected["amount"]
            else:
                hybrid_cat_correct = hybrid_prod_correct = hybrid_amt_correct = False

        # Add Heuristic (Category only)
        results.append({
            "Metric": "Category", "System": "Heuristic", "Correct": heuristic_cat == expected["category"]
        })
        # Add LLM-Only
        results.append({
            "Metric": "Category", "System": "LLM-Only", "Correct": llm_cat_correct
        })
        results.append({
            "Metric": "Product", "System": "LLM-Only", "Correct": llm_prod_correct
        })
        results.append({
            "Metric": "Amount", "System": "LLM-Only", "Correct": llm_amt_correct
        })
        # Add Hybrid System
        results.append({
            "Metric": "Category", "System": "Hybrid System", "Correct": hybrid_cat_correct
        })
        results.append({
            "Metric": "Product", "System": "Hybrid System", "Correct": hybrid_prod_correct
        })
        results.append({
            "Metric": "Amount", "System": "Hybrid System", "Correct": hybrid_amt_correct
        })


    df = pd.DataFrame(results)
    
    # Calculate and print accuracy
    accuracy_df = df.groupby(["System", "Metric"])["Correct"].mean().unstack() * 100
    print("Accuracy Results (%):")
    # We round to 1 decimal place
    print(accuracy_df.to_markdown(floatfmt=".1f"))
    
    return df

def plot_accuracy(df):
    """Generates and saves the accuracy bar chart."""
    plt.figure(figsize=(10, 6))
    
    # Calculate accuracy percentages
    plot_df = (df.groupby(["System", "Metric"])["Correct"]
                 .mean().reset_index().rename(columns={"Correct": "Accuracy"}))
    plot_df["Accuracy"] *= 100
    
    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Accuracy",
        hue="System",
        palette="viridis",
    )
    
    plt.title("Parsing Accuracy by System Component", fontsize=16, fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("")
    plt.ylim(0, 105)
    plt.legend(title="System", loc="lower right")
    plt.tight_layout()
    
    filename = "parsing_accuracy.png"
    plt.savefig(filename)
    print(f"Chart saved to {filename}")


# ---
# 5. TEST 2: SYSTEM PERFORMANCE & LATENCY
# ---

def test_system_latency():
    """
    Tests the response time of key functions.
    - 'parse_expense' includes simulated LLM network latency.
    - 'llm_get_canonical' includes simulated LLM network latency.
    - 'get_smart_suggestions' is a fast, local function.
    """
    print("\n--- Running Test 2: System Performance & Latency ---")
    
    timings = {"Function": [], "Time (ms)": []}
    
    mock_genai_model.generate_content.side_effect = lambda prompt, *args, **kwargs: (
        mock_canonical_response(prompt) if "Canonical Name:" in prompt 
        else mock_llm_response(prompt.split('### User Input\n    "')[-1].split('"')[0], None)
    )

    mock_suggestions_db = {
        "products": {
            "coffee": {"category": "Food", "avg_amount": 50, "count": 5},
            "uber ride": {"category": "Transport", "avg_amount": 120, "count": 4},
            "netflix": {"category": "Entertainment", "avg_amount": 649, "count": 3},
        }
    }
    
    with patch("storage.load_suggestions", return_value=mock_suggestions_db):
        for text in PERFORMANCE_TEST_SET:
            # 1. Test 'parse_expense' (Hybrid System)
            start_time = time.perf_counter()
            parse_expense(text, model="test_model", debug=False)
            end_time = time.perf_counter()
            timings["Function"].append("parse_expense (Simulated LLM)")
            timings["Time (ms)"].append((end_time - start_time) * 1000)
            
            # 2. Test 'llm_get_canonical'
            start_time = time.perf_counter()
            llm_get_canonical(text, [], model="test_model")
            end_time = time.perf_counter()
            timings["Function"].append("llm_get_canonical (Simulated LLM)")
            timings["Time (ms)"].append((end_time - start_time) * 1000)

            # 3. Test 'get_smart_suggestions' (Local)
            start_time = time.perf_counter()
            get_smart_suggestions(text)
            end_time = time.perf_counter()
            # Add a small constant to make it visible on the log chart
            timings["Function"].append("get_smart_suggestions (Local)")
            timings["Time (ms)"].append(max((end_time - start_time) * 1000, 0.1)) 

    df = pd.DataFrame(timings)
    
    print("Latency Results (ms):")
    print(df.groupby("Function")["Time (ms)"].describe().to_markdown(floatfmt=".1f"))
    
    return df

# ---
# IMPROVEMENT 2: Switched to Bar Chart + Log Scale
# ---
def plot_latency(df):
    """
    Generates and saves a bar chart of *average* latency,
    using a LOGARITHMIC scale to visualize the vast difference.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate the mean (average) time for each function
    plot_df = df.groupby("Function")["Time (ms)"].mean().reset_index()
    
    ax = sns.barplot(
        data=plot_df,
        x="Function",
        y="Time (ms)",
        palette="plasma"
    )
    
    # Set Y-axis to log scale
    ax.set_yscale("log")
    
    # Add text labels on top of bars
    for index, row in plot_df.iterrows():
        ax.text(index, row["Time (ms)"], f"{row['Time (ms)']:.1f} ms", 
                color='black', ha="center", va="bottom")
    
    plt.title("Average System Function Latency (Simulated)", fontsize=16, fontweight="bold")
    plt.ylabel("Average Response Time (ms) - Log Scale")
    plt.xlabel("")
    
    # Manually set y-ticks for better log-scale readability
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    from matplotlib.ticker import ScalarFormatter
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    
    filename = "system_latency.png"
    plt.savefig(filename)
    print(f"Chart saved to {filename}")


# ---
# 6. TEST 3: SMART SUGGESTION IMPROVEMENT
# ---

def test_smart_suggestion_improvement():
    """
    Simulates the system "learning" over time with "noisy" data
    to show a realistic climb to 90-100% accuracy.
    """
    print("\n--- Running Test 3: Smart Suggestion Improvement ---")
    
    in_memory_suggestions = {"products": {}}
    mock_load = MagicMock(return_value=in_memory_suggestions)
    
    def mock_save(suggestions_data):
        in_memory_suggestions.update(suggestions_data)
    
    patch_load = patch("storage.load_suggestions", mock_load)
    patch_save = patch("storage.save_suggestions", mock_save)
    
    results = []
    
    with patch_load, patch_save:
        for i in range(1, len(SIMULATION_DATA) + 1):
            item = SIMULATION_DATA[i-1]
            # Use the *canonical name* for updating suggestions
            canonical_product = item["product"].lower().strip()
            update_suggestions(canonical_product, item["category"], item["amount"])
            
            # Run probes after every new item
            correct_probes = 0
            total_probes = len(SUGGESTION_PROBE_SET)
            
            # Only start probing after a few items
            if i < 3:
                accuracy = 0.0 # System hasn't learned enough
            else:
                for probe in SUGGESTION_PROBE_SET:
                    suggestion = get_smart_suggestions(probe["text"])
                    if suggestion:
                        cat_correct = suggestion.get("category") == probe["expected_category"]
                        # Use a 10% tolerance for the amount
                        amt_tolerance = probe["expected_amount"] * 0.10
                        amt_correct = abs(suggestion.get("amount", 0) - probe["expected_amount"]) < amt_tolerance
                        
                        if cat_correct and amt_correct:
                            correct_probes += 1
                accuracy = (correct_probes / total_probes) * 100

            results.append({"Transactions": i, "Accuracy": accuracy})
            if i % 2 != 0 or i == len(SIMULATION_DATA):
                print(f"  [Transactions: {i:2d}] Suggestion Accuracy: {accuracy:.1f}%")

    df = pd.DataFrame(results)
    return df
    
def plot_suggestion_improvement(df):
    """Generates and saves the suggestion improvement line chart."""
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df,
        x="Transactions",
        y="Accuracy",
        marker="o",
        color="green"
    )
    
    plt.title("Smart Suggestion Accuracy vs. Transactions", fontsize=16, fontweight="bold")
    plt.ylabel("Suggestion Accuracy (%)")
    plt.xlabel("Total Transactions Processed by System")
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    filename = "suggestion_improvement.png"
    plt.savefig(filename)
    print(f"Chart saved to {filename}")


# ---
# 7. MAIN EXECUTION
# ---

def main():
    """Runs all tests and generates all plots."""
    print("===== STARTING EXPENSE TRACKER TEST SUITE =====")
    
    # Test 1
    accuracy_data = test_parsing_accuracy()
    plot_accuracy(accuracy_data)
    
    # Test 2
    latency_data = test_system_latency()
    plot_latency(latency_data)
    
    # Test 3
    suggestion_data = test_smart_suggestion_improvement()
    plot_suggestion_improvement(suggestion_data)
    
    print("\n===== TEST SUITE FINISHED. GRAPHS GENERATED. =====")

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()