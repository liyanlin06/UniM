import openai
import json
import os

# ===== Configuration =====
openai.api_key = os.getenv("OPENAI_API_KEY")

# ===== Input: example HTML snippet =====
code_md = """```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Styled Page</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: space-between;
            height: 100vh;
            background-color: #f9f9f9;
        }

        header {
            margin-top: 20px;
        }

        header img {
            width: 100px;
        }

        main {
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: left;
            max-width: 800px;
            padding: 20px;
        }

        .image-placeholder {
            width: 300px;
            height: 200px;
            background-color: #ccc;
            margin-right: 20px;
        }

        .description {
            max-width: 400px;
        }

        footer {
            display: flex;
            justify-content: space-around;
            width: 100%;
            padding: 10px 0;
            background-color: #fff;
        }

        footer a {
            text-decoration: none;
            color: blue;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <header>
        <img src="rick.jpg" alt="Logo">
    </header>
    <main>
        <div class="image-placeholder"></div>
        <div class="description">
            <p>Explore the world of glamour and confidence with our premium collections. Every piece has been lovingly crafted in our hand-picked and luxury materials.</p>
        </div>
    </main>
    <footer>
        <a href="#">Home</a>
        <a href="#">About</a>
        <a href="#">Contact</a>
    </footer>
</body>
</html>
```"""

# ===== Prompt design =====
prompt = f"""
You are a strict code reviewer.
You will evaluate a piece of code written in Markdown format.

The evaluation must be language-agnostic.
Do NOT assume specific language rules; judge only based on general software engineering best practices.

## Evaluation Dimensions (each scored 0-100):
1. correctness: Syntax soundness, logic consistency, handling of edge cases, absence of obvious errors.
2. readability: Clarity of naming, structure, comments, lack of redundancy or magic values.
3. design: Modularity, separation of concerns, clear input/output contracts, maintainability.
4. performance: Avoidance of unnecessary complexity, loops, recursion, or inefficient patterns.
5. security: Avoid obvious risks (e.g. injection, unsafe handling, secrets in code).
6. testability: Code can be tested; boundaries and error cases observable; minimal hidden global state.

## Requirements:
- Output ONLY valid JSON, nothing else.
- Do NOT wrap the JSON in markdown code blocks (no ```json or ```).
- Do NOT add any text before or after the JSON.
- The JSON must be a single object with these exact keys: "correctness", "readability", "design", "performance", "security", "testability", "overall_score"
- Each score must be an integer from 0 to 100.
- "overall_score" must be the average of all six dimensions, rounded to the nearest integer.
- Example format: {{"correctness": 85, "readability": 90, "design": 80, "performance": 85, "security": 75, "testability": 70, "overall_score": 81}}

Now evaluate the following code:

{code_md}   
"""

# ===== Call OpenAI =====
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)

# ===== Parse output =====
output_text = response.choices[0].message.content.strip()
print("Raw Output:\n", output_text)

try:
    result = json.loads(output_text)
    print("\n=== Parsed Result ===")
    print(json.dumps(result, indent=4))
    print("\nFinal Overall Score:", result.get("overall_score"))
except Exception as e:
    print("Failed to parse JSON:", e)
