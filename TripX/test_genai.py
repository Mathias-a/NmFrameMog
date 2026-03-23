import asyncio

from google import genai

API_KEY = "AIzaSyDXWbMGEpkYrJUpf-qArxglOZrN56GTnr8"
client = genai.Client(api_key=API_KEY)

async def main():
    response = await client.aio.models.generate_content(
        model='gemini-3.1-pro-preview',
        contents='Tell me a joke'
    )
    print(response.text)

asyncio.run(main())
