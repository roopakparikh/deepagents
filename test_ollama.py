import requests
import json

class OllamaLLM:
    def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def _call(self, prompt):
        # Construct the API request
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False  # Disable streaming for simplicity
        }

        # Send the request
        response = requests.post(url, data=json.dumps(data), headers={"Content-Type": "application/json"})
        response.raise_for_status()

        # Parse the response
        result = response.json()
        return result["response"]

    def generate(self, prompt, stop=None):
        # Generate text using the Ollama model
        return self._call(prompt)

    @property
    def name(self):
        return self.model_name


if __name__ == "__main__":
    ollama = OllamaLLM('qwen3:4b')
    prompt = """
    You are a helpful assistant that can use the following tools to answer questions:

{
  "tools": [
    {
      "name": "get_weather_by_zip",
      "description": "Fetches weather data for a given zip code.",
      "parameters": {
        "zip_code": { "type": "string", "description": "The zip code for the location." }
      }
    }
  ]
}

When the user asks for weather information, use this tool to retrieve the data.

User: what is the weather for zip code 95120
    """
    out = ollama.generate(prompt)
    print(out)