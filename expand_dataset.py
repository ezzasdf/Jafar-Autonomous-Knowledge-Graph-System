"""
Expand reasoning_dataset.jsonl with 25 beginner Python Q&A entries.

Topics: File I/O, error handling, classes, modules, pip, stdlib modules.

Usage:
    python expand_dataset.py
"""
import hashlib
import json
import random
import time
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(BASE, "data", "reasoning_dataset.jsonl")

NEW_ENTRIES = [
    # ---- File I/O (5) ----
    {
        "problem": "How do I open and read a file in Python?",
        "reasoning": "Use the built-in open() function with a context manager (with statement) which auto-closes the file, even if an error occurs. Specify the mode ('r' for read) and encoding.",
        "answer": "with open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()",
        "tags": ["file_io", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "What's the difference between 'w', 'a', and 'r+' file modes?",
        "reasoning": "'w' overwrites the file, 'a' appends to the end, 'r+' opens for both reading and writing without truncating. Choose based on whether you need to preserve existing content.",
        "answer": "'w' — write (truncates), 'a' — append (preserves), 'r+' — read+write (no truncate). Use 'w' for new output, 'a' for logs, 'r+' for updating in place.",
        "tags": ["file_io", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How do I write to a file in Python?",
        "reasoning": "Open the file in write ('w') or append ('a') mode inside a with block, then call f.write() or f.writelines(). The with statement ensures the file is closed properly.",
        "answer": "with open('output.txt', 'w', encoding='utf-8') as f:\n    f.write('Hello, world!')",
        "tags": ["file_io", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How do I read a file line by line in Python?",
        "reasoning": "Use a for loop directly on the file object, which yields one line at a time without loading the entire file into memory. This is memory-efficient for large files.",
        "answer": "with open('file.txt', 'r') as f:\n    for line in f:\n        print(line.strip())",
        "tags": ["file_io", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How do I check if a file exists before reading it?",
        "reasoning": "Use os.path.isfile() or pathlib.Path.exists() to check without opening. This lets you handle missing files gracefully instead of crashing with FileNotFoundError.",
        "answer": "import os\nif os.path.exists('file.txt'):\n    with open('file.txt') as f:\n        print(f.read())",
        "tags": ["file_io", "beginner"],
        "domain": "learning",
    },
    # ---- Error handling (5) ----
    {
        "problem": "How does try/except work in Python?",
        "reasoning": "Wrap risky code in the try block. If an exception occurs, execution jumps to the matching except block. You can catch specific exception types or use a bare except.",
        "answer": "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')",
        "tags": ["error_handling", "beginner"],
        "domain": "debugging",
    },
    {
        "problem": "What's the difference between except, else, and finally?",
        "reasoning": "except runs when an exception is caught, else runs when NO exception occurred, finally ALWAYS runs (even on exceptions or return). Use finally for cleanup like closing files.",
        "answer": "try:\n    x = int(input())\nexcept ValueError:\n    print('Bad input')\nelse:\n    print(f'You entered {x}')\nfinally:\n    print('This always runs')",
        "tags": ["error_handling", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How do I raise my own exceptions in Python?",
        "reasoning": "Use the 'raise' keyword followed by an exception instance or class. You can raise built-in exceptions or custom ones to signal error conditions intentionally.",
        "answer": "raise ValueError('Invalid value provided')",
        "tags": ["error_handling", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How do I create a custom exception class?",
        "reasoning": "Inherit from Exception (or a subclass) to create a custom exception. This lets you raise and catch domain-specific errors with meaningful names.",
        "answer": "class MyCustomError(Exception):\n    pass\n\nraise MyCustomError('Something went wrong')",
        "tags": ["error_handling", "beginner"],
        "domain": "design",
    },
    {
        "problem": "How do I catch multiple exception types in one except block?",
        "reasoning": "Parenthesize the exception types in a tuple. The except block catches any of them and you can access the instance with 'as e'.",
        "answer": "try:\n    x = int('abc')\n    y = 10 / 0\nexcept (ValueError, ZeroDivisionError) as e:\n    print(f'Caught: {e}')",
        "tags": ["error_handling", "beginner"],
        "domain": "debugging",
    },
    # ---- Classes (5) ----
    {
        "problem": "How do I define a class in Python?",
        "reasoning": "Use the 'class' keyword. The __init__ method initializes instances. 'self' refers to the instance itself and is the first parameter of all instance methods.",
        "answer": "class Dog:\n    def __init__(self, name):\n        self.name = name\n\n    def bark(self):\n        return f'{self.name} says woof!'",
        "tags": ["classes", "oop", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "What is 'self' in Python class methods?",
        "reasoning": "'self' is the conventional name for the instance reference. It's automatically passed when calling a method on an instance. It gives access to instance attributes and other methods.",
        "answer": "self refers to the current instance. It's how methods access the object's own data. Always the first parameter, but never passed explicitly when calling.",
        "tags": ["classes", "oop", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How does inheritance work in Python?",
        "reasoning": "Pass the parent class as a parameter in the class definition. The child inherits all methods and attributes. Use super() to call the parent's __init__ or methods.",
        "answer": "class Animal:\n    def __init__(self, name):\n        self.name = name\n\nclass Dog(Animal):\n    def __init__(self, name, breed):\n        super().__init__(name)\n        self.breed = breed",
        "tags": ["classes", "oop", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "What's the difference between @staticmethod and @classmethod?",
        "reasoning": "@staticmethod doesn't receive self or cls — it's just a function in the class namespace. @classmethod receives cls (the class) and can modify class state. Use staticmethod for utility functions, classmethod for alternative constructors.",
        "answer": "class MyClass:\n    @staticmethod\n    def util(x): return x * 2\n\n    @classmethod\n    def create(cls, data): return cls(data)  # factory method",
        "tags": ["classes", "oop", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How do I use @property decorators in Python?",
        "reasoning": "Property decorators turn method calls into attribute access. Use @property for getter, @x.setter for setter, @x.deleter for deletion. This enables computed attributes without breaking backward compatibility.",
        "answer": "class Circle:\n    def __init__(self, radius):\n        self._radius = radius\n\n    @property\n    def area(self):\n        return 3.14 * self._radius ** 2\n\n    @property\n    def radius(self):\n        return self._radius\n\n    @radius.setter\n    def radius(self, value):\n        if value < 0:\n            raise ValueError('Radius must be positive')\n        self._radius = value",
        "tags": ["classes", "oop", "beginner"],
        "domain": "design",
    },
    # ---- Modules (4) ----
    {
        "problem": "How do I import a module in Python?",
        "reasoning": "Use 'import module_name' to import the whole module, or 'from module import name' for specific items. 'import module as alias' renames it for convenience.",
        "answer": "import math\nprint(math.sqrt(16))\n\nfrom os import listdir\nprint(listdir('.'))\n\nimport numpy as np",
        "tags": ["modules", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "What does 'if __name__ == '__main__':' do?",
        "reasoning": "This guard checks whether the file is run directly or imported. Code inside runs only when the script is executed, not when imported as a module. Essential for creating reusable libraries.",
        "answer": "def main():\n    print('Running directly')\n\nif __name__ == '__main__':\n    main()  # Only runs when script is executed, not imported",
        "tags": ["modules", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "What is __init__.py used for?",
        "reasoning": "__init__.py marks a directory as a Python package. It can define __all__ to control what's exported with 'from package import *', and runs package-level initialization code.",
        "answer": "# mypackage/__init__.py\n__all__ = ['module_a', 'module_b']\nprint('mypackage loaded')",
        "tags": ["modules", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "What is the difference between a module and a package in Python?",
        "reasoning": "A module is a single .py file. A package is a directory containing an __init__.py file, which can contain multiple modules. Packages can be nested.",
        "answer": "A module is a .py file. A package is a directory with __init__.py that groups related modules together, enabling hierarchical imports like 'from package.submodule import func'.",
        "tags": ["modules", "beginner"],
        "domain": "learning",
    },
    # ---- pip commands (3) ----
    {
        "problem": "How do I install a Python package with pip?",
        "reasoning": "Use 'pip install package_name' from the command line. Add --user to install for the current user only, or -U to upgrade. Virtual environments isolate packages per project.",
        "answer": "pip install requests\npip install numpy==1.24.0\npip install --upgrade pip\npip install -r requirements.txt",
        "tags": ["pip", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How do I list all installed Python packages?",
        "reasoning": "Use 'pip list' to see all installed packages and versions. 'pip freeze' shows them in requirements.txt format which is useful for reproducing environments.",
        "answer": "pip list          # Table format\npip freeze        # requirements.txt format\npip freeze > requirements.txt  # Save current environment",
        "tags": ["pip", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How do I uninstall a Python package?",
        "reasoning": "Use 'pip uninstall package_name'. It prompts for confirmation unless you add -y. Multiple packages can be listed. Can't uninstall packages that other packages depend on.",
        "answer": "pip uninstall requests\npip uninstall -y numpy  # Skip confirmation",
        "tags": ["pip", "beginner"],
        "domain": "learning",
    },
    # ---- Common stdlib modules (3) ----
    {
        "problem": "What are the most useful stdlib modules for beginners?",
        "reasoning": "os for file/path operations, sys for interpreter access, json for data serialization, datetime for dates, re for regex, math/random for numeric work, collections for advanced data structures, pathlib for modern path handling.",
        "answer": "import os, sys, json, datetime, re, math, random\nfrom collections import defaultdict, Counter\nfrom pathlib import Path",
        "tags": ["stdlib", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How do I work with JSON in Python?",
        "reasoning": "Use json.loads() to parse a JSON string into Python objects, or json.load() to read from a file. json.dumps() converts Python objects to JSON strings, json.dump() writes to a file.",
        "answer": "import json\n\ndata = {\"name\": \"Alice\", \"age\": 30}\njson_str = json.dumps(data)\nparsed = json.loads(json_str)\n\nwith open('data.json', 'w') as f:\n    json.dump(data, f)",
        "tags": ["stdlib", "json", "beginner"],
        "domain": "learning",
    },
    {
        "problem": "How do I work with dates and times in Python?",
        "reasoning": "Use the datetime module. datetime.datetime.now() gets current time, .strftime() formats, .strptime() parses strings. timedelta handles date arithmetic.",
        "answer": "from datetime import datetime, timedelta\n\nnow = datetime.now()\nprint(now.strftime('%Y-%m-%d %H:%M:%S'))\n\nyesterday = now - timedelta(days=1)\ndt = datetime.strptime('2024-01-01', '%Y-%m-%d')",
        "tags": ["stdlib", "datetime", "beginner"],
        "domain": "learning",
    },
]


def make_embedding(text: str, dim: int = 384) -> list[float]:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    vec = [rng.uniform(-0.1, 0.1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5
    return [round(x / norm, 6) for x in vec]


def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)

    existing_ids = set()
    if os.path.isfile(DST):
        with open(DST, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass

    added = 0
    with open(DST, "a", encoding="utf-8") as f:
        for entry in NEW_ENTRIES:
            raw = f"{entry['problem']}|{entry['reasoning']}"
            ex_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
            if ex_id in existing_ids:
                continue
            existing_ids.add(ex_id)

            text_for_emb = f"{entry['problem']} {entry['reasoning']} {entry['answer']}"
            record = {
                "id": ex_id,
                "problem": entry["problem"],
                "reasoning": entry["reasoning"],
                "answer": entry["answer"],
                "tags": entry["tags"],
                "domain": entry["domain"],
                "confidence": 0.7,
                "usage_count": 0,
                "success_count": 0,
                "created_at": time.time(),
                "embedding": make_embedding(text_for_emb),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            added += 1

    total = len(existing_ids)
    print(f"Added {added} new entries. Dataset now has {total} total entries.")


if __name__ == "__main__":
    main()
