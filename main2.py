import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://ai-bootcamp-openai-pod4.openai.azure.com/"
model_name = "text-embedding-3-small"
deployment = "text-embedding-3-small"

api_version = "2024-02-01"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=endpoint,
)

response = client.embeddings.create(
    input=["first phrase","second phrase","third phrase"],
    model=deployment
)

for item in response.data:
    length = len(item.embedding)
    print(
        f"data[{item.index}]: length={length}, "
        f"[{item.embedding[0]}, {item.embedding[1]}, "
        f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
    )
print(response.usage)