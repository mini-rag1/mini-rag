from ..LLMInterface import LLMInterface
from ..LLMEnums import CohereEnums, DocumentTypeEnum
from cohere import ClientV2
import logging
from typing import List,Union

class CohereProvider(LLMInterface):
    def __init__(self, api_key: str,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        self.api_key = api_key
        self.client = ClientV2(api_key)

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self,model_id:str):
        self.generation_model_id = model_id

    def set_embedding_model(self,model_id:str,embedding_size:int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None, temperature: float = None):
        if not self.client:
            self.logger.error("Cohere client not initialized.")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model ID not set.")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(self.construct_prompt(prompt=prompt, role=CohereEnums.USER.value))

        response = self.client.chat(
            model=self.generation_model_id,
            messages=chat_history,
            max_tokens=max_output_tokens,
            temperature=temperature
        )

        # Log the full response to inspect its structure
        self.logger.debug(f"Full chat response: {response}")

        # Extract the generated text from the response
        if not response or not response.message or not response.message.content:
            self.logger.error(f"Error in generation response or no content returned. Response: {response}")
            return None

        # Access the first content item and its text
        generated_text = response.message.content[0].text if response.message.content else None

        if not generated_text:
            self.logger.error(f"No text found in the response content. Response: {response}")
            return None

        return generated_text

    def process_text(self,text:str):
        return text[:self.default_input_max_characters].strip()
        
    
    def embed_text(self, text:Union[str,List[str]], document_type:str):
        if not self.client:
            self.logger.error("Cohere client not initialized.")
            return None

        if isinstance(text, str):
            text = [text]

        if not self.embedding_model_id:
            self.logger.error("Embedding model ID not set.")
            return None
        
        input_type = CohereEnums.DOCUMENT.value
        if document_type == DocumentTypeEnum.QUERY.value:
            input_type = CohereEnums.QUERY.value

        # Add retry logic with exponential backoff
        max_retries = 5
        base_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                response = self.client.embed(
                    model=self.embedding_model_id,
                    texts=[self.process_text(t) for t in text],
                    input_type=input_type,
                    embedding_types=["float"]
                )
                
                if not response or not response.embeddings or not response.embeddings.float_:
                    self.logger.error("Error while embedding text with Cohere.")
                    return None
                
                return [f for f in response.embeddings.float]
                
            except Exception as e:
                # Check if it's a rate limit error
                if hasattr(e, 'status_code') and e.status_code == 429:
                    # Calculate exponential backoff with jitter
                    import random
                    import time
                    
                    delay = (2 ** attempt) * base_delay + random.uniform(0, 1)
                    self.logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Error embedding text: {str(e)}")
                    return None
        
        self.logger.error(f"Failed to embed text after {max_retries} attempts due to rate limiting")
        return None
    
    def construct_prompt(self, prompt: str, role: str):
        # Ensure the role is lowercase to match Cohere's API requirements
        return {
            "role": role.lower(),  # Convert role to lowercase
            "content": prompt
        }