#parser.py
import json
import re
import time
import os

api_key = os.environ.get("GOOGLE_API_KEY")

import google.generativeai as genai
genai.configure(api_key=api_key)

import streamlit as st
import random
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
from collections import Counter

from config import (
    CANON_CATEGORIES, CATEGORY_KEYWORDS, STOPWORDS, 
    INAPPROPRIATE_CATEGORIES, MODERATION_PROMPT_TEMPLATE
)
from storage import get_all_categories, add_custom_category

# -----------------------------
# Enhanced Category Descriptions
# -----------------------------
CATEGORY_DESCRIPTIONS = {
    "Food": "Restaurant meals, takeout, fast food, snacks, beverages, coffee, tea",
    "Groceries": "Supermarket items, vegetables, fruits, milk, bread, eggs, cooking ingredients",
    "Transport": "Bus, metro, train, taxi, Uber, Ola, auto, fuel, petrol, parking",
    "Stationery": "Pens, pencils, notebooks, paper, office supplies, printing",
    "Entertainment": "Movies, streaming services, games, concerts, music, shows, hobbies",
    "Utilities": "Electricity, water, gas, internet, phone bills, recharges",
    "Health": "Medicine, doctor visits, hospital, pharmacy, medical supplies, vitamins",
    "Education": "Tuition, courses, books, exam fees, classes, training",
    "Personal": "Toiletries, cosmetics, grooming, haircare, skincare, hygiene products",
    "Bills": "Rent, maintenance, subscriptions, recurring payments",
    "Electronics": "Phones, laptops, computers, gadgets, chargers, accessories",
    "Home & Appliances": "Furniture, appliances, home improvement, kitchenware",
    "Services": "Repairs, cleaning, consulting, professional services, maintenance",
    "Other": "Items that don't fit other categories"
}

# -----------------------------
# Enhanced Keyword Matching
# -----------------------------
def get_enhanced_keywords():
    """Return enhanced keyword mappings with more comprehensive coverage."""
    enhanced = CATEGORY_KEYWORDS.copy()
    
    enhanced["Food"].extend(["restaurant", "cafe", "dhaba", "biryani", "dosa", "samosa", 
                             "ice cream", "juice", "sandwich", "noodles", "pasta"])
    enhanced["Groceries"].extend(["dal", "wheat", "vegetables", "onion", "potato", "tomato",
                                  "spices", "masala", "atta", "grocery store"])
    enhanced["Transport"].extend(["rapido", "travel", "commute", "ride", "booking", "ticket fare"])
    enhanced["Entertainment"].extend(["theatre", "gaming", "subscription", "ott", "streaming"])
    enhanced["Health"].extend(["pharmacy", "checkup", "consultation", "tablet", "syrup", "injection"])
    
    return enhanced

# -----------------------------
# Fuzzy Category Matching
# -----------------------------
def fuzzy_match_category(text: str, threshold: float = 0.75) -> Optional[str]:
    """Match text to categories using fuzzy string matching."""
    text_lower = text.lower()
    all_categories = get_all_categories()
    
    best_match = None
    best_score = 0.0
    
    for category in all_categories:
        similarity = SequenceMatcher(None, text_lower, category.lower()).ratio()
        
        if category.lower() in text_lower or text_lower in category.lower():
            similarity = max(similarity, 0.8)
        
        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = category
    
    return best_match

# -----------------------------
# Historical Context Analyzer (FIXED)
# -----------------------------
def get_historical_category(product: str) -> Optional[str]:
    """Get category based on historical data for similar products."""
    try:
        from storage import load_suggestions
        suggestions = load_suggestions()
        products_data = suggestions.get("products", {})
        
        if not products_data:
            return None
        
        # Check for exact match
        product_lower = product.lower()
        if product_lower in products_data:
            return products_data[product_lower]["category"]
        
        # Check for partial matches
        for stored_product, data in products_data.items():
            if product_lower in stored_product or stored_product in product_lower:
                if data.get("count", 0) >= 2:
                    return data["category"]
        
        return None
    except Exception as e:
        st.warning(f"Historical category lookup failed: {e}")
        return None

# -----------------------------
# Heuristic & Regex Parsing
# -----------------------------
def regex_amount(text: str) -> List[int]:
    """Extract numerical amounts from text using regex."""
    patterns = [
        r"(?i)(?:₹|rs\.?|rupees?|rps|bucks|rupee)\s*(\d+)",
        r"(\d+)\s*(?:₹|rs\.?|rupees?|rps|bucks|rupee)",
        r"\b(\d+)\b"
    ]
    nums = []
    for p in patterns:
        matches = re.finditer(p, text)
        nums.extend([int(m.group(1)) for m in matches])
    return nums

def parse_amount(text: str) -> int:
    """Get the most likely amount from text."""
    n_list = regex_amount(text)
    return n_list[-1] if n_list else 0

def normalize_category(cat: str) -> str:
    """Normalize a category string with enhanced fuzzy matching."""
    if not cat: return "Other"
    c = cat.strip()
    
    mapping = {
        "snack": "Food", "snacks": "Food", "meal": "Food", "dining": "Food",
        "grocery": "Groceries", "supermarket": "Groceries",
        "travel": "Transport", "commute": "Transport", "ride": "Transport",
        "movie": "Entertainment", "movies": "Entertainment",
        "medical": "Health", "medicine": "Health", "healthcare": "Health",
        "school": "Education", "college": "Education",
        "clothing": "Personal", "clothes": "Personal",
        "tech": "Electronics", "gadget": "Electronics",
        "furniture": "Home & Appliances", "appliance": "Home & Appliances",
        "service": "Services", "repair": "Services"
    }
    
    c_lower = c.lower()
    if c_lower in mapping:
        return mapping[c_lower]
    
    all_categories = get_all_categories()
    for category in all_categories:
        if c_lower == category.lower():
            return category
    
    fuzzy_match = fuzzy_match_category(c)
    if fuzzy_match:
        return fuzzy_match
    
    return "Other"

def keyword_category(text: str) -> str:
    """Enhanced keyword-based category guessing with scoring."""
    t = text.lower()
    enhanced_keywords = get_enhanced_keywords()
    
    scores = Counter()
    
    for cat, words in enhanced_keywords.items():
        for word in words:
            pattern = rf"\b{re.escape(word)}\b"
            matches = len(re.findall(pattern, t))
            if matches > 0:
                scores[cat] += matches
    
    if scores:
        return scores.most_common(1)[0][0]
    
    return "Other"

def heuristic_item(text: str) -> str:
    """Extract a plausible item/product name from text."""
    t = re.sub(r"(?i)(₹|rs\.?|rupees?)\s*\d+", " ", text)
    t = re.sub(r"\b\d+\b", " ", t)
    toks = [w for w in re.findall(r"[a-zA-Z]+", t.lower()) if w not in STOPWORDS]
    return " ".join(toks[:4]).title() if toks else "Item"

# -----------------------------
# Smart Suggestions (FIXED & INTEGRATED)
# -----------------------------
def get_smart_suggestions(text: str) -> Dict:
    """
    Get smart suggestions based on matching products in input text.
    Returns the best match with category and average amount.
    """
    try:
        from storage import load_suggestions
        
        suggestions = load_suggestions()
        products_data = suggestions.get("products", {})
        
        if not products_data:
            return {}
        
        text_lower = text.lower().strip()
        text_words = set(text_lower.split())
        
        matches = []
        for product, data in products_data.items():
            product_lower = product.lower()
            product_words = set(product_lower.split())
            
            # Strategy 1: Exact match (highest priority)
            if text_lower == product_lower:
                return {
                    "product": product,
                    "category": data["category"],
                    "amount": data["avg_amount"],
                    "confidence": data["count"],
                    "match_score": 1.0,
                    "match_type": "exact"
                }
            
            # Strategy 2: Text contains product or vice versa
            if product_lower in text_lower or text_lower in product_lower:
                match_score = min(len(product_lower), len(text_lower)) / max(len(product_lower), len(text_lower))
                matches.append({
                    "product": product,
                    "category": data["category"],
                    "amount": data["avg_amount"],
                    "confidence": data["count"],
                    "match_score": match_score * 0.9,  # Slightly lower than exact
                    "match_type": "substring"
                })
                continue
            
            # Strategy 3: Word overlap
            overlap = text_words.intersection(product_words)
            if overlap:
                # Calculate match score based on overlap
                match_score = len(overlap) / len(product_words)
                
                matches.append({
                    "product": product,
                    "category": data["category"],
                    "amount": data["avg_amount"],
                    "confidence": data["count"],
                    "match_score": match_score,
                    "match_type": "word_overlap"
                })
        
        if not matches:
            return {}
        
        # Sort by match score first, then by confidence (count)
        matches.sort(key=lambda x: (x["match_score"], x["confidence"]), reverse=True)
        
        best_match = matches[0]
        
        # More lenient threshold: accept if >40% match OR high confidence
        if best_match["match_score"] >= 0.4 or (best_match["match_score"] >= 0.3 and best_match["confidence"] >= 5):
            return best_match
        
        return {}
        
    except Exception as e:
        if st.session_state.get("debug", False):
            st.warning(f"Smart suggestions failed: {e}")
        return {}

def get_category_suggestions(text: str, model: str) -> List[Dict]:
    """Get multiple category suggestions with confidence scores from the LLM."""
    all_categories = get_all_categories()
    
    cat_descriptions = "\n".join([f"- {cat}: {CATEGORY_DESCRIPTIONS.get(cat, '')}" 
                                   for cat in all_categories])
    
    prompt = f"""
You are a professional expense categorization assistant.

Analyze the following text and predict the 3 most likely expense categories it belongs to.

### Available Categories with Descriptions:
{cat_descriptions}

### Rules
- Use only categories from the list above
- Output must be **strict JSON only**, no explanations outside JSON
- Each entry must contain:
  - "category": a single category name
  - "confidence": integer 0–100 (reflecting probability)
  - "reason": short 1-sentence justification

### Examples
Text: "Ordered pizza for dinner"
JSON: [
  {{"category": "Food", "confidence": 95, "reason": "Pizza is a food item typically ordered from restaurants"}},
  {{"category": "Groceries", "confidence": 20, "reason": "Could be ingredients if homemade"}},
  {{"category": "Entertainment", "confidence": 10, "reason": "Dining out can be leisure activity"}}
]

Text: "Bought milk and bread"
JSON: [
  {{"category": "Groceries", "confidence": 98, "reason": "Milk and bread are common grocery items"}},
  {{"category": "Food", "confidence": 15, "reason": "Could be immediate consumption items"}},
  {{"category": "Other", "confidence": 5, "reason": "Fallback category"}}
]

### Task
Text: "{text}"
JSON:
"""
    
    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )
        suggestions = json.loads(response.text.strip())
        return suggestions if isinstance(suggestions, list) else [suggestions]
    except Exception:
        return [{"category": "Other", "confidence": 40, "reason": "LLM categorization failed"}]

# -----------------------------
# Custom Category Parsing
# -----------------------------
def parse_custom_category(text: str) -> Tuple[Optional[str], Optional[str], str]:
    """Parse text for custom category assignment using the 'as' keyword."""
    as_pattern = r'(.+?)\s+as\s+([a-zA-Z][a-zA-Z\s]*?)(?:\s+for\s+|\s+)(\d+.*)'
    match = re.search(as_pattern, text.lower())
    
    if match:
        item = match.group(1).strip().title()
        category = match.group(2).strip().title()
        remaining_text = match.group(3).strip()
        return item, category, remaining_text
    
    return None, None, text

# -----------------------------
# Product Consistency
# -----------------------------
def find_similar_products(product1: str, product_list: List[str], threshold: float = 0.8) -> List[Tuple[str, float]]:
    """Finds products in a list that are similar to a given product using SequenceMatcher."""
    similarities = []
    p1_lower = product1.lower()
    for product2 in product_list:
        p2_lower = product2.lower()
        if p1_lower == p2_lower:
            continue
        
        similarity = SequenceMatcher(None, p1_lower, p2_lower).ratio()
        if similarity >= threshold:
            similarities.append((product2, similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)

# -----------------------------
# Multi-Stage Category Determination (WITH SMART SUGGESTIONS)
# -----------------------------
def determine_best_category(text: str, product: str, llm_category: str, model: str) -> Tuple[str, int]:
    """
    Use multiple strategies to determine the best category and amount.
    Returns (category, suggested_amount)
    Priority: Smart Suggestions > Historical > LLM > Keywords > Fuzzy Match
    """
    # Stage 0: Smart Suggestions (NEW - HIGHEST PRIORITY)
    smart_match = get_smart_suggestions(text)
    if smart_match and smart_match.get("confidence", 0) >= 3:
        # If we've seen this product 3+ times, trust it
        return smart_match["category"], smart_match["amount"]
    
    # Stage 1: Check historical data
    historical = get_historical_category(product)
    if historical and historical != "Other":
        # Get amount from smart suggestions if available
        suggested_amount = smart_match.get("amount", 0) if smart_match else 0
        return historical, suggested_amount
    
    # Stage 2: Trust LLM if confident
    if llm_category and llm_category != "Other":
        return llm_category, 0
    
    # Stage 3: Enhanced keyword matching
    keyword_cat = keyword_category(text)
    if keyword_cat != "Other":
        return keyword_cat, 0
    
    # Stage 4: Fuzzy category matching
    fuzzy_cat = fuzzy_match_category(product, threshold=0.7)
    if fuzzy_cat:
        return fuzzy_cat, 0
    
    # Stage 5: LLM suggestions
    suggestions = get_category_suggestions(text, model)
    if suggestions and suggestions[0].get("confidence", 0) > 60:
        return normalize_category(suggestions[0]["category"]), 0
    
    return "Other", 0

# -----------------------------
# Core LLM Parsing Functions
# -----------------------------
def safe_json_extract(raw_text: str) -> Dict:
    """Safely extract a JSON object or array from a string, handling markdown code blocks."""
    # Remove markdown code blocks if present
    cleaned = raw_text.strip()
    
    # Remove ```json and ``` markers
    if cleaned.startswith("```"):
        # Find the actual JSON content between code blocks
        lines = cleaned.split('\n')
        json_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block or (not line.strip().startswith("```") and json_lines):
                json_lines.append(line)
        
        cleaned = '\n'.join(json_lines).strip()
    
    # Try direct JSON parsing first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object {...}
    obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array [...]
    arr_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', cleaned, re.DOTALL)
    if arr_match:
        try:
            result = json.loads(arr_match.group(0))
            # Wrap array in object if needed for consistency
            if isinstance(result, list):
                return {"items": result}
            return result
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"No valid JSON found in text: {cleaned[:200]}...")

max_retries = 3

def llm_parse_with_retry(text: str, model: str, debug: bool = False) -> Dict:
    """Parse a single expense item using an LLM with enhanced prompts and retry logic."""
    all_categories = get_all_categories()
    
    cat_descriptions = "\n".join([f"  - {cat}: {CATEGORY_DESCRIPTIONS.get(cat, '')}" 
                                   for cat in all_categories])
    
    prompt = f"""
You are an intelligent expense parser specialized in Indian retail and services.

### Objective
Extract these three fields from the input text:
1. "product" – short, clear noun phrase describing the item/service
2. "category" – ONE category from the list below
3. "amount" – integer amount in rupees (no symbols)

### Available Categories:
{cat_descriptions}

### Output Format
Return **strict JSON only**:
{{"product": "...", "category": "...", "amount": 0}}

### Guidelines
- NO explanations or text outside JSON
- "product" should be concise (2-4 words max)
- Choose the MOST SPECIFIC category that fits
- Only use "Other" if genuinely unclear
- Amount must be a plain integer (50, not ₹50)
- Consider Indian context (e.g., "chai" is tea/coffee in Food)

### Examples
Input: "bought coffee from cafe for 80rs"
Output: {{"product": "coffee", "category": "Food", "amount": 80}}

Input: "uber ride to office 120"
Output: {{"product": "uber ride", "category": "Transport", "amount": 120}}

Input: "notebook and pen 45 rupees"
Output: {{"product": "notebook and pen", "category": "Stationery", "amount": 45}}

### Input
"{text}"
### JSON:
""".strip()

    for attempt in range(max_retries):
        try:
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(
                prompt,
                generation_config={"temperature": 0}
            )
            raw = response.text.strip()
            if debug: 
                st.code(f"Attempt {attempt + 1}: {raw}", language="json")
            
            data = safe_json_extract(raw)
            return {
                "product": (data.get("product") or "item").strip(),
                "category": normalize_category(data.get("category", "Other")),
                "amount": int(float(data.get("amount", 0)))
            }
        except Exception as e:
            if debug: 
                st.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                if debug: 
                    st.error("All LLM attempts for single item failed.")
                return {"product": "item", "category": "Other", "amount": 0}

def detect_multiple_items(text: str, model: str, debug: bool = False) -> Tuple[bool, List[Dict]]:
    """Detect and parse multiple items with enhanced prompts and better error handling."""
    separators = [",", " and ", "&", "+", " also ", " plus "]
    has_separators = any(sep in text.lower() for sep in separators)
    amount_patterns = re.findall(r'\d+\s*(?:rs\.?|rupees?|₹)', text.lower())
    multiple_amounts = len(amount_patterns) > 1

    if not (has_separators or multiple_amounts):
        return False, []

    all_categories = get_all_categories()
    cat_list = ", ".join(all_categories)
    
    prompt = f"""
You are an expert expense parser.

### Task
Analyze if the text contains one or multiple expense items.

### Rules
1. ONE item → return: {{"multiple": false}}
2. MULTIPLE items → return:
   {{
     "multiple": true,
     "items": [
       {{"product": "...", "category": "...", "amount": 0}},
       ...
     ]
   }}
3. Each item needs:
   - "product": concise noun phrase (2-4 words)
   - "category": from {cat_list}
   - "amount": integer only

### Examples
Text: "coffee for 20rs and notebook for 10rs"
→ {{"multiple": true, "items": [{{"product": "coffee", "category": "Food", "amount": 20}}, {{"product": "notebook", "category": "Stationery", "amount": 10}}]}}

Text: "fruits 200rs, notebooks 60rs"
→ {{"multiple": true, "items": [{{"product": "fruits", "category": "Groceries", "amount": 200}}, {{"product": "notebooks", "category": "Stationery", "amount": 60}}]}}

Text: "coffee 50"
→ {{"multiple": false}}

Text: "uber 150, sandwich 80, water bottle 20"
→ {{"multiple": true, "items": [{{"product": "uber ride", "category": "Transport", "amount": 150}}, {{"product": "sandwich", "category": "Food", "amount": 80}}, {{"product": "water bottle", "category": "Groceries", "amount": 20}}]}}

### CRITICAL
- Return ONLY valid JSON, no explanations
- Do NOT include markdown code blocks
- Separate each item with its own amount

### Input
"{text}"
### JSON:
"""
    
    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json"  # Force JSON response
            }
        )
        raw_response = response.text.strip()
        
        if debug:
            st.code(f"Multi-item raw response:\n{raw_response}", language="json")
        
        result = safe_json_extract(raw_response)
        
        if debug:
            st.json(result)

        if result.get("multiple") and "items" in result and isinstance(result["items"], list):
            valid_items = []
            for item in result["items"]:
                if "product" in item and "amount" in item:
                    item["category"] = normalize_category(item.get("category", "Other"))
                    item["amount"] = int(item.get("amount", 0))
                    valid_items.append(item)
            
            if valid_items:
                if debug:
                    st.success(f"✅ Found {len(valid_items)} items")
                return True, valid_items

    except json.JSONDecodeError as e:
        if debug:
            st.error(f"JSON parsing failed: {e}")
            st.code(raw_response, language="text")
        st.warning(f"Multi-item parsing failed (JSON error). Falling back to single item mode.")
        return False, []
    except Exception as e:
        if debug:
            st.error(f"Multi-item parsing error: {e}")
        st.warning(f"Multi-item parsing failed: {str(e)}. Falling back to single item mode.")
        return False, []

    return False, []

def check_for_inappropriate_content(text: str, model: str) -> str | None:
    """Uses the LLM to classify text content."""
    categories_str = ", ".join(INAPPROPRIATE_CATEGORIES)
    
    prompt = f"""
Classify the following text into one of these moderation categories:
{categories_str}

### Rules
- Respond with **only** the category name (lowercase, no punctuation)
- Choose "safe" if the text is not offensive or inappropriate
- Do not include explanations or reasoning

Text: "{text}"
Category:
"""

    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            prompt,
            generation_config={"temperature": 0.1}
        )
        category = response.text.strip().lower().replace(" ", "_")

        if category in INAPPROPRIATE_CATEGORIES and category != "safe":
            return category
        
        return None
    except Exception as e:
        st.warning(f"Content moderation check failed: {e}")
        return None

def llm_get_canonical(product_name: str, existing_names: List[str], model: str) -> str:
    """Uses the LLM to generate a canonical (standard) name for a product."""
    
    examples = ", ".join(random.sample(existing_names, min(len(existing_names), 5))) if existing_names else "None"
    
    prompt = f"""
You are a data normalization specialist.

### Goal
Return the simplest, most standard name (canonical form) for the given product.

### Guidelines
- Remove personal or temporal words (my, special, morning, today)
- Keep essential descriptors (filter, iced, whole wheat, etc.)
- Correct spelling errors
- Convert to lowercase
- If a similar term exists in these examples → prefer that form: {examples}
- Output only the canonical name, nothing else

### Examples
"My Morning Coffee" → "coffee"
"Filter Coffee" → "filter coffee"
"Uber ride home" → "uber ride"

Product: "{product_name}"
Canonical Name:
"""

    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(
            prompt,
            generation_config={"temperature": 0}
        )
        return response.text.strip().lower()
    except Exception as e:
        st.warning(f"Canonical name generation failed: {e}")
        return product_name.lower().strip()

# -----------------------------
# Main Hybrid Parser (FIXED)
# -----------------------------
def parse_expense(text: str, model: str, debug: bool = False) -> Tuple[bool, List[Dict], str]:
    """
    Enhanced parser with smart suggestions and improved categorization.
    """
    text = (text or "").strip()
    if not text:
        return False, [], "Input is empty."

    # 1. Content Moderation
    inappropriate_category = check_for_inappropriate_content(text, model)
    if inappropriate_category:
        reason = inappropriate_category.replace("_", " ").title()
        return False, [], f"Inappropriate content detected ({reason}). Please be respectful."

    # 2. Custom Category Parsing
    custom_item, custom_category, remaining_text = parse_custom_category(text)
    if custom_item and custom_category:
        if add_custom_category(custom_category):
            st.toast(f"New category created: '{custom_category}'")
        
        amount = parse_amount(remaining_text)
        return True, [{"product": custom_item, "category": custom_category, "amount": amount}], ""

    # 3. Multi-Item Parsing
    is_multiple, items = detect_multiple_items(text, model, debug)
    if is_multiple:
        processed_items = []
        for item in items:
            if isinstance(item, dict) and item.get("product") and isinstance(item.get("amount"), int):
                best_cat, suggested_amt = determine_best_category(
                    text, 
                    item["product"], 
                    item.get("category", "Other"),
                    model
                )
                item["category"] = best_cat
                # Use suggested amount if current amount is 0
                if item["amount"] == 0 and suggested_amt > 0:
                    item["amount"] = suggested_amt
                processed_items.append(item)
        if processed_items:
            if debug:
                st.success(f"✅ Successfully parsed {len(processed_items)} items")
            return True, processed_items, ""

    # 4. Single-Item Parsing with Smart Suggestions
    llm_result = llm_parse_with_retry(text, model, debug)
    heuristic_result = {
        "product": heuristic_item(text),
        "category": keyword_category(text),
        "amount": parse_amount(text)
    }

    # Combine results
    final_product = llm_result["product"] if llm_result["product"] not in ["item", "Item"] else heuristic_result["product"]
    
    # CRITICAL: Check if user provided an amount in the input
    user_provided_amount = llm_result["amount"] > 0 or heuristic_result["amount"] > 0
    
    # Try smart suggestions to get category AND amount
    smart_match = get_smart_suggestions(text)
    
    if smart_match and smart_match.get("confidence", 0) >= 2:
        # We have a good historical match!
        if debug:
            st.info(f"✨ Smart suggestion found: {smart_match['product']} → {smart_match['category']} (₹{smart_match['amount']})")
        
        best_category = smart_match["category"]
        suggested_amount = smart_match["amount"]
    else:
        # No smart match, use multi-stage determination
        best_category, suggested_amount = determine_best_category(
            text,
            final_product,
            llm_result["category"],
            model
        )
    
    # FIXED: Prioritize user's explicit amount over suggestions
    final_amount = 0
    if user_provided_amount:
        # User provided an amount - ALWAYS use it (respect user input)
        final_amount = llm_result["amount"] if llm_result["amount"] > 0 else heuristic_result["amount"]
        if debug and suggested_amount > 0:
            st.info(f"ℹ️ Using your amount (₹{final_amount}) instead of suggested (₹{suggested_amount})")
    else:
        # No amount in input - use smart suggestion
        if suggested_amount > 0:
            final_amount = suggested_amount
            if debug:
                st.success(f"💡 Auto-filled amount: ₹{final_amount} (from history)")
    
    final_item = {
        "product": final_product,
        "category": best_category,
        "amount": final_amount
    }

    # Amount validation: Only enforce if user didn't provide amount AND no smart suggestion
    if final_item["amount"] <= 0:
        if smart_match:
            # Smart match exists but amount is still 0 (shouldn't happen, but handle it)
            return False, [], "⚠️ Smart suggestion found but amount is invalid. Please include an amount."
        else:
            # No smart match and no amount in input
            return False, [], "⚠️ Please include a monetary amount (e.g., 'rent 15000' or 'coffee 50rs')\n\n💡 Tip: After adding this expense once with an amount, you can use just the product name next time!"

    return True, [final_item], ""