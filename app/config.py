#v2
# config.py
# Central configuration file for the Smart Expense Tracker application.

# File paths
CSV_FILE = "expenses.csv"
BUDGET_FILE = "budgets.json"
SUGGESTIONS_FILE = "suggestions.json"
PRODUCT_ALIASES_FILE = "product_aliases.json"
CATEGORIES_FILE = "custom_categories.json"

# LLM Configuration
DEFAULT_MODEL = "gemini-2.5-flash"

# Application constants
CANON_CATEGORIES = [
    "Food", "Groceries", "Transport", "Stationery", "Entertainment",
    "Utilities", "Health", "Education", "Personal", "Bills", 
    "Electronics", "Home & Appliances", "Services", "Other"
]

# Enhanced keywords for heuristic-based category guessing
# More comprehensive coverage with Indian context
CATEGORY_KEYWORDS = {
    "Food": [
        # Meals & dining
        "meal", "lunch", "dinner", "breakfast", "brunch",
        # Fast food & snacks
        "chips", "snack", "snacks", "pizza", "burger", "fries", "sandwich", "samosa", "vada", "pakora",
        # Beverages
        "tea", "coffee", "chai", "juice", "shake", "lassi", "soda", "cold drink",
        # Restaurants & outlets
        "restaurant", "cafe", "dhaba", "canteen", "mess", "food court", "mcd", "kfc", "dominos",
        # Indian food
        "biryani", "dosa", "idli", "paratha", "roti", "naan", "curry", "dal", "paneer",
        # Desserts
        "ice cream", "cake", "pastry", "sweet", "mithai", "chocolate"
    ],
    
    "Groceries": [
        # Staples
        "rice", "flour", "atta", "wheat", "dal", "pulses", "sugar", "salt", "oil",
        # Produce
        "milk", "bread", "egg", "eggs", "vegetable", "vegetables", "fruit", "fruits",
        "onion", "potato", "tomato", "green", "leafy",
        # Packaged goods
        "grocery", "groceries", "supermarket", "kirana", "provisions",
        # Cooking
        "spices", "masala", "cooking", "baking"
    ],
    
    "Transport": [
        # Public transport
        "bus", "metro", "train", "railway", "ticket", "fare", "pass",
        # Ride services
        "uber", "ola", "rapido", "auto", "rickshaw", "cab", "taxi", "ride",
        # Fuel
        "petrol", "diesel", "fuel", "gas", "refuel", "filling",
        # Parking & tolls
        "parking", "toll", "travel", "commute", "transport"
    ],
    
    "Stationery": [
        # Writing supplies
        "pen", "pencil", "marker", "highlighter", "crayon",
        # Paper products
        "notebook", "book", "diary", "notepad", "paper", "sheet",
        # Office supplies
        "folder", "file", "stapler", "clip", "tape", "glue",
        "eraser", "sharpener", "ruler", "scissors"
    ],
    
    "Entertainment": [
        # Streaming & subscriptions
        "netflix", "prime", "hotstar", "spotify", "youtube", "ott", "streaming", "subscription",
        # Events
        "movie", "cinema", "theatre", "theater", "show", "concert", "event",
        # Games & hobbies
        "game", "gaming", "sport", "music", "hobby", "club"
    ],
    
    "Utilities": [
        # Services
        "electricity", "power", "water", "gas", "lpg", "cylinder",
        # Internet & communication
        "wifi", "internet", "broadband", "data", "recharge", "prepaid", "postpaid",
        "mobile bill", "phone bill", "airtel", "jio", "vodafone", "vi"
    ],
    
    "Health": [
        # Medical
        "medicine", "tablet", "syrup", "capsule", "injection", "prescription",
        "doctor", "clinic", "hospital", "pharmacy", "medical", "checkup", "consultation",
        # Healthcare products
        "mask", "sanitizer", "bandage", "gauze", "thermometer",
        "vitamin", "supplement", "first aid", "health"
    ],
    
    "Education": [
        # Fees & courses
        "tuition", "course", "class", "training", "workshop", "seminar",
        "exam", "fee", "fees", "admission", "registration",
        # Institutions
        "school", "college", "university", "institute", "academy",
        # Materials
        "lecture", "study", "textbook", "reference", "tutorial", "coaching"
    ],
    
    "Personal": [
        # Hygiene
        "soap", "shampoo", "conditioner", "toothpaste", "toothbrush", "mouthwash",
        "tissue", "towel", "napkin",
        # Grooming
        "perfume", "deodorant", "cologne", "fragrance",
        "razor", "shaving", "trimmer", "haircut", "salon", "parlor",
        # Beauty & skincare
        "cream", "lotion", "moisturizer", "sunscreen", "facewash",
        "cosmetic", "makeup", "skincare", "haircare", "bodycare"
    ],
    
    "Bills": [
        # Recurring payments
        "rent", "maintenance", "society", "association",
        "subscription", "membership", "bill", "invoice",
        "emi", "installment", "payment", "due"
    ],
    
    "Electronics": [
        # Devices
        "phone", "mobile", "smartphone", "iphone", "android",
        "laptop", "computer", "desktop", "pc", "tablet", "ipad",
        # Peripherals
        "mouse", "keyboard", "monitor", "screen", "display",
        "printer", "scanner", "webcam", "speaker", "headphone", "earphone", "earbuds",
        # Accessories
        "charger", "cable", "adapter", "battery", "powerbank",
        "case", "cover", "protector",
        # Tech
        "tv", "television", "camera", "smartwatch", "gadget", "electronic", "tech"
    ],
    
    "Home & Appliances": [
        # Major appliances
        "fridge", "refrigerator", "washing machine", "microwave", "oven",
        "ac", "air conditioner", "cooler", "heater", "geyser",
        # Kitchen appliances
        "mixer", "grinder", "toaster", "kettle", "blender", "juicer",
        # Furniture
        "furniture", "table", "chair", "sofa", "couch", "bed", "mattress",
        "cupboard", "wardrobe", "shelf", "rack",
        # Home items
        "lamp", "light", "bulb", "fan", "curtain", "carpet", "rug",
        "utensil", "cookware", "crockery", "kitchenware",
        "vacuum", "cleaner", "iron", "appliance", "home"
    ],
    
    "Services": [
        # Professional services
        "repair", "fix", "maintenance", "service", "servicing",
        "cleaning", "laundry", "washing", "ironing",
        "plumber", "electrician", "carpenter", "painter",
        # Digital services
        "consulting", "consultation", "professional", "expert",
        "subscription", "hosting", "domain", "support",
        "delivery", "shipping", "courier", "installation"
    ]
}

# Stopwords for cleaning up product names
STOPWORDS = set([
    "bought", "buy", "got", "for", "on", "at", "of", "the", "a", "an", 
    "some", "to", "my", "i", "as", "from", "paid", "spend", "spent",
    "purchased", "purchase", "today", "yesterday", "just", "went"
])

# -----------------------------
# Content Moderation
# -----------------------------

# The LLM will classify input into one of these categories. "safe" is the only acceptable category.
INAPPROPRIATE_CATEGORIES = [
    "safe", "abusive", "hate_speech", "sexual_content", "self_harm", "violence"
]

# Prompt template for the content moderation check.
MODERATION_PROMPT_TEMPLATE = """
You are a content moderation expert. Your task is to classify the user's text into one of the following predefined categories: {categories_str}.
- Respond with ONLY the category name and nothing else.
- Be strict in your classification. If the text is even remotely offensive or inappropriate, classify it accordingly.
- If the text is a normal, everyday expense entry, classify it as "safe".

Text to classify: "{text}"
Category:
"""