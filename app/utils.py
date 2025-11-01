#v2
#utils.py
import streamlit as st
import speech_recognition as sr
import pandas as pd
import io
import random
import datetime as dt
from datetime import timedelta

from config import CANON_CATEGORIES

def capture_speech() -> str:
    """
    Captures audio from the microphone and transcribes it to text.
    Handles common errors gracefully.
    """
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Listening… (Auto-stops on silence)")
            r.adjust_for_ambient_noise(source, duration=0.5)
            # Increased timeout to give users more time to speak
            audio = r.listen(source, timeout=10, phrase_time_limit=10)
        
        with st.spinner("Transcribing…"):
            text = r.recognize_google(audio)
        st.success(f"🗣️ Heard: \"{text}\"")
        return text
        
    except sr.WaitTimeoutError:
        st.warning("Listening timeout. No speech detected.")
    except sr.RequestError as e:
        st.error(f"Speech API request failed: {e}. Check internet connection.")
    except sr.UnknownValueError:
        st.error("Could not understand the audio. Please try speaking clearly.")
    except Exception as e:
        st.error(f"A microphone error occurred: {e}")
    return ""

def export_data(df: pd.DataFrame, format_type: str) -> bytes:
    """
    Export a DataFrame to the specified format (CSV, JSON, or Excel).
    Includes a summary sheet for Excel exports.
    """
    if format_type == "CSV":
        return df.to_csv(index=False).encode('utf-8')
    elif format_type == "JSON":
        return df.to_json(orient="records", date_format="iso").encode('utf-8')
    elif format_type == "Excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Expenses', index=False)
            
            # Add a summary sheet with category breakdowns
            summary = df.groupby('category')['amount'].agg(['sum', 'count', 'mean']).round(2)
            summary.to_excel(writer, sheet_name='Summary')
            
        return buffer.getvalue()

def generate_sample_data() -> pd.DataFrame:
    """
    Generate realistic sample expense entries with patterns for smart suggestions.
    Creates 300 entries over the last 3 months with repeated products at similar prices.
    """
    sample_data = []
    start_date = dt.datetime.now() - timedelta(days=90)
    
    # Enhanced product catalog with realistic Indian products and variations
    products_by_category = {
        "Food": [
            # Coffee variations (will test smart suggestions)
            ("Coffee", 50), ("Coffee", 60), ("Coffee", 55), ("Coffee", 50), ("Coffee", 60),
            ("Cappuccino", 80), ("Cappuccino", 85), ("Cappuccino", 80),
            ("Filter Coffee", 40), ("Filter Coffee", 45), ("Filter Coffee", 40),
            
            # Other food items with consistent pricing
            ("Pizza", 350), ("Pizza", 400), ("Pizza", 380), ("Pizza", 350),
            ("Biryani", 200), ("Biryani", 220), ("Biryani", 210), ("Biryani", 200),
            ("Burger", 150), ("Burger", 160), ("Burger", 150),
            ("Dosa", 80), ("Dosa", 90), ("Dosa", 85),
            ("Sandwich", 100), ("Sandwich", 120), ("Sandwich", 110),
            ("Chai", 20), ("Chai", 25), ("Chai", 20), ("Chai", 20),
            ("Samosa", 15), ("Samosa", 20), ("Samosa", 15),
            ("Ice Cream", 60), ("Ice Cream", 70), ("Ice Cream", 65),
        ],
        "Groceries": [
            ("Milk", 60), ("Milk", 65), ("Milk", 60), ("Milk", 60),
            ("Bread", 40), ("Bread", 45), ("Bread", 40), ("Bread", 42),
            ("Eggs", 80), ("Eggs", 85), ("Eggs", 80),
            ("Rice", 500), ("Rice", 550), ("Rice", 520),
            ("Vegetables", 200), ("Vegetables", 250), ("Vegetables", 220),
            ("Fruits", 300), ("Fruits", 350), ("Fruits", 320),
            ("Atta", 400), ("Atta", 420), ("Atta", 410),
            ("Dal", 150), ("Dal", 160), ("Dal", 155),
            ("Oil", 200), ("Oil", 220), ("Oil", 210),
        ],
        "Transport": [
            ("Uber Ride", 120), ("Uber Ride", 150), ("Uber Ride", 130), ("Uber Ride", 140),
            ("Ola Ride", 100), ("Ola Ride", 110), ("Ola Ride", 105),
            ("Auto Rickshaw", 50), ("Auto Rickshaw", 60), ("Auto Rickshaw", 55),
            ("Bus Ticket", 30), ("Bus Ticket", 35), ("Bus Ticket", 30),
            ("Metro Ticket", 40), ("Metro Ticket", 50), ("Metro Ticket", 45),
            ("Petrol", 500), ("Petrol", 600), ("Petrol", 550),
        ],
        "Entertainment": [
            ("Movie Ticket", 300), ("Movie Ticket", 350), ("Movie Ticket", 320),
            ("Netflix", 649), ("Netflix", 649), ("Netflix", 649),
            ("Prime Video", 299), ("Prime Video", 299),
            ("Spotify", 119), ("Spotify", 119), ("Spotify", 119),
            ("Gaming", 500), ("Gaming", 600), ("Gaming", 550),
        ],
        "Stationery": [
            ("Notebook", 50), ("Notebook", 60), ("Notebook", 55),
            ("Pen", 20), ("Pen", 25), ("Pen", 20),
            ("Pencil", 10), ("Pencil", 15), ("Pencil", 12),
            ("Marker", 30), ("Marker", 35), ("Marker", 30),
        ],
        "Utilities": [
            ("Mobile Recharge", 299), ("Mobile Recharge", 399), ("Mobile Recharge", 299),
            ("WiFi Bill", 799), ("WiFi Bill", 799), ("WiFi Bill", 799),
            ("Electricity Bill", 1500), ("Electricity Bill", 1800), ("Electricity Bill", 1600),
        ],
        "Health": [
            ("Medicine", 150), ("Medicine", 200), ("Medicine", 180),
            ("Doctor Visit", 500), ("Doctor Visit", 600), ("Doctor Visit", 550),
            ("Vitamins", 300), ("Vitamins", 350), ("Vitamins", 320),
        ],
        "Personal": [
            ("Shampoo", 250), ("Shampoo", 280), ("Shampoo", 260),
            ("Toothpaste", 80), ("Toothpaste", 90), ("Toothpaste", 85),
            ("Soap", 40), ("Soap", 50), ("Soap", 45),
            ("Haircut", 200), ("Haircut", 250), ("Haircut", 220),
        ],
        "Bills": [
            ("Rent", 15000), ("Rent", 15000), ("Rent", 15000),
            ("Maintenance", 2000), ("Maintenance", 2000), ("Maintenance", 2000),
        ]
    }
    
    # Generate entries with realistic patterns
    entry_id = 1
    for category, products in products_by_category.items():
        for product, amount in products:
            # Generate 1-2 entries for each product-amount pair
            for _ in range(random.randint(1, 2)):
                date = start_date + timedelta(days=random.randint(0, 90))
                time_hour = random.randint(8, 22)
                time_minute = random.randint(0, 59)
                
                # Add slight variation to amounts (±10%) to make it realistic
                varied_amount = int(amount * random.uniform(0.95, 1.05))
                
                sample_data.append({
                    "id": entry_id,
                    "date": date.date(),
                    "time": f"{time_hour:02d}:{time_minute:02d}:00",
                    "product": product,
                    "canonical_product": product.lower(),  # Add canonical product
                    "category": category,
                    "amount": varied_amount
                })
                entry_id += 1
    
    # Shuffle to make it realistic
    random.shuffle(sample_data)
    
    # Reassign IDs in order
    for idx, entry in enumerate(sample_data, start=1):
        entry["id"] = idx
    
    return pd.DataFrame(sample_data)


def get_smart_suggestion_test_cases():
    """
    Returns a list of test cases to demonstrate the smart suggestions feature.
    Each test case includes the input text and expected behavior.
    """
    test_cases = [
        {
            "input": "coffee 55",
            "expected": "Should suggest 'Coffee' in Food category with amount around ₹50-60",
            "explanation": "The system has seen 'Coffee' multiple times at similar prices"
        },
        {
            "input": "morning cappuccino",
            "expected": "Should suggest 'Cappuccino' in Food category with amount around ₹80-85",
            "explanation": "Cappuccino appears consistently in the data"
        },
        {
            "input": "uber to office",
            "expected": "Should suggest 'Uber Ride' in Transport category with amount around ₹120-150",
            "explanation": "Uber rides are frequently recorded in the data"
        },
        {
            "input": "bought milk",
            "expected": "Should suggest 'Milk' in Groceries category with amount around ₹60",
            "explanation": "Milk is a common grocery item with consistent pricing"
        },
        {
            "input": "netflix subscription",
            "expected": "Should suggest 'Netflix' in Entertainment category with amount ₹649",
            "explanation": "Netflix has a fixed subscription price"
        },
        {
            "input": "chai at chai point",
            "expected": "Should suggest 'Chai' in Food category with amount around ₹20-25",
            "explanation": "Chai (tea) is frequently purchased at low prices"
        },
        {
            "input": "monthly rent payment",
            "expected": "Should suggest 'Rent' in Bills category with amount ₹15,000",
            "explanation": "Rent is a recurring monthly expense"
        },
        {
            "input": "auto rickshaw ride",
            "expected": "Should suggest 'Auto Rickshaw' in Transport category with amount around ₹50-60",
            "explanation": "Auto rides are common short-distance transport"
        },
        {
            "input": "biryani from restaurant",
            "expected": "Should suggest 'Biryani' in Food category with amount around ₹200-220",
            "explanation": "Biryani is a popular food item with consistent pricing"
        },
        {
            "input": "movie at pvr",
            "expected": "Should suggest 'Movie Ticket' in Entertainment category with amount around ₹300-350",
            "explanation": "Movie tickets have relatively stable pricing"
        }
    ]
    
    return test_cases


def print_test_cases():
    """Print test cases for smart suggestions feature."""
    test_cases = get_smart_suggestion_test_cases()
    
    print("\n" + "="*80)
    print("SMART SUGGESTIONS TEST CASES")
    print("="*80)
    print("\nAfter loading sample data, try these inputs to test smart suggestions:\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"{i}. INPUT: '{test['input']}'")
        print(f"   EXPECTED: {test['expected']}")
        print(f"   WHY: {test['explanation']}\n")
    
    print("="*80)
    print("\nNOTE: Smart suggestions work by:")
    print("1. Matching product names in your input with historical data")
    print("2. Finding the average amount spent on that product")
    print("3. Suggesting the category and amount based on past patterns")
    print("="*80 + "\n")