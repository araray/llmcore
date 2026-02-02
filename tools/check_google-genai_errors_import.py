import google.genai as genai

print(genai.__version__)

from google.genai import errors

client = genai.Client()
try:
    client.models.generate_content(
        model="invalid-model-name",
        contents="What is your name?",
    )
except errors.APIError as e:
    print(e.code)  # 404
    print(e.message)
