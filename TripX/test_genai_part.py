from google.genai import types

part = types.Part.from_function_response(
    name="execute_tripletex_api",
    response={"result": "ok"}
)
print(part)
