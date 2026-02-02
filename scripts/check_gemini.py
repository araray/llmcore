from os import environ

from google import genai

client = genai.Client(api_key=environ["GEMINI_API_KEY"])
models = list(client.models.list())
nmodels = len(models)
print(nmodels)
print(models)
