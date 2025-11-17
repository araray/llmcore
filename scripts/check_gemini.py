from google import genai
from os import environ
client = genai.Client(api_key=environ["GEMINI_API_KEY"])
models = list(client.models.list())
nmodels = len(models)
print(nmodels)
print(models)
