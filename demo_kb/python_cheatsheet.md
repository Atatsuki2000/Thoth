# Python Quick Reference

## Data Types

### Basic Types
```python
# Numbers
integer = 42
floating = 3.14
complex_num = 2 + 3j

# Strings
text = "Hello World"
multiline = """Multiple
lines"""

# Boolean
is_true = True
is_false = False

# None
nothing = None
```

### Collections
```python
# List (mutable, ordered)
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")  # Add item
fruits[0]  # Access by index: "apple"

# Tuple (immutable, ordered)
coordinates = (10, 20)
x, y = coordinates  # Unpacking

# Set (mutable, unordered, unique)
unique_nums = {1, 2, 3, 2}  # {1, 2, 3}
unique_nums.add(4)

# Dictionary (key-value pairs)
person = {"name": "Alice", "age": 30}
person["email"] = "alice@example.com"
```

## Control Flow

### Conditionals
```python
if temperature > 30:
    print("It's hot!")
elif temperature > 20:
    print("It's warm")
else:
    print("It's cold")

# Ternary operator
status = "adult" if age >= 18 else "minor"
```

### Loops
```python
# For loop
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

for fruit in fruits:
    print(fruit)

# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
```

## Functions

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Default arguments
def power(base, exponent=2):
    return base ** exponent

# Variable arguments
def sum_all(*args):
    return sum(args)

# Keyword arguments
def create_profile(**kwargs):
    return kwargs

# Lambda (anonymous function)
square = lambda x: x**2
```

## Classes and Objects

```python
class Dog:
    # Class variable
    species = "Canis familiaris"
    
    # Constructor
    def __init__(self, name, age):
        self.name = name  # Instance variable
        self.age = age
    
    # Instance method
    def bark(self):
        return f"{self.name} says Woof!"
    
    # String representation
    def __str__(self):
        return f"{self.name} is {self.age} years old"

# Create instance
my_dog = Dog("Buddy", 3)
print(my_dog.bark())  # "Buddy says Woof!"
```

## File Operations

```python
# Reading files
with open("file.txt", "r") as f:
    content = f.read()  # Read entire file
    # lines = f.readlines()  # List of lines

# Writing files
with open("output.txt", "w") as f:
    f.write("Hello World\n")

# Appending
with open("log.txt", "a") as f:
    f.write("New log entry\n")
```

## Common Libraries

### os - Operating System
```python
import os
os.getcwd()  # Current directory
os.listdir(".")  # List files
os.path.exists("file.txt")  # Check if exists
```

### datetime
```python
from datetime import datetime, timedelta

now = datetime.now()
tomorrow = now + timedelta(days=1)
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
```

### json
```python
import json

# Dictionary to JSON
data = {"name": "Alice", "age": 30}
json_str = json.dumps(data)

# JSON to dictionary
parsed = json.loads(json_str)
```

### requests (HTTP)
```python
import requests

response = requests.get("https://api.example.com/data")
data = response.json()
```

## Error Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
else:
    print("No errors occurred")
finally:
    print("Always executes")
```

## Common String Methods

```python
text = "  Hello World  "

text.strip()  # Remove whitespace: "Hello World"
text.lower()  # Lowercase: "  hello world  "
text.upper()  # Uppercase: "  HELLO WORLD  "
text.replace("World", "Python")  # "  Hello Python  "
text.split()  # ["Hello", "World"]
"-".join(["a", "b", "c"])  # "a-b-c"
"Hello" in text  # True
```

## Useful Built-in Functions

```python
# Type conversions
int("42")  # 42
str(42)  # "42"
float("3.14")  # 3.14
list("abc")  # ['a', 'b', 'c']

# Iterables
len([1, 2, 3])  # 3
sum([1, 2, 3])  # 6
max([1, 2, 3])  # 3
sorted([3, 1, 2])  # [1, 2, 3]
reversed([1, 2, 3])  # [3, 2, 1] (iterator)

# Others
print("Hello")
input("Enter name: ")
type(42)  # <class 'int'>
isinstance(42, int)  # True
```

## Virtual Environments

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install packages
pip install requests

# Save dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Deactivate
deactivate
```
