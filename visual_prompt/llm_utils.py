"""
GPT-4o API integration for visual prompting.
This module provides functions for encoding images, preparing prompts with text and images,
and handling requests to the OpenAI GPT-4 Vision API with support for in-context examples.
"""

import os
import numpy as np
import base64
import requests
from io import BytesIO
from typing import List, Union, Optional, Dict, Any
from PIL import Image

# Get OpenAI API Key from environment variable
openai_api_key = os.environ['OPENAI_API_KEY']

API_URL = "https://api.openai.com/v1/chat/completions"


def encode_image_to_base64(image: Union[str, Image.Image, np.ndarray]) -> str:
    """
    Encodes an image into a base64-encoded string in JPEG format.

    Args:
        image (Union[str, Image.Image, np.ndarray]): The image to be encoded. This can be a string
            of the image path, a PIL image, or a numpy array.

    Returns:
        str: A base64-encoded string representing the image in JPEG format.
        
    Raises:
        ValueError: If the image type is not supported.
    """
    # Function to encode the image
    def _encode_image_from_file(image_path: str) -> str:
        # Function to encode the image
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    def _encode_image_from_pil(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format='JPEG')
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    if isinstance(image, str):
        return _encode_image_from_file(image)
    elif isinstance(image, Image.Image):
        return _encode_image_from_pil(image)
    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
        return _encode_image_from_pil(image_pil)
    else:
        raise ValueError(f"Unknown option for image {type(image)}")


def prepare_prompt(
    images: List[Union[Image.Image, np.ndarray]],
    prompt: Optional[str] = None,
    in_context_examples: Optional[List[Dict[str, Any]]] = None
  ) -> Dict[str, Any]:
  """
  Prepares a prompt with images and text for the GPT API.
  
  Args:
      images (List[Union[Image.Image, np.ndarray]]): List of images to include in the prompt.
      prompt (Optional[str]): Text prompt to accompany the images. Defaults to None.
      in_context_examples (Optional[List[Dict[str, Any]]]): List of in-context examples, 
          where each example is a dict with 'images', 'prompt', and 'response' keys. Defaults to None.
  
  Returns:
      Dict[str, Any]: A formatted prompt dictionary ready to be sent to the GPT API.
      
  Raises:
      AssertionError: If both images and text prompts are empty.
  """
  def _append_pair(current_prompt: Dict[str, Any], images: List[Union[Image.Image, np.ndarray]], text: Optional[str]) -> Dict[str, Any]:
    # text first if given, then image.
    if text:
      current_prompt['content'].append({
          'type': 'text',
          'text': text
        })
    else:
      assert len(images) > 0, "Both images and text prompts are empty."

    for image in images:
      base64_image = encode_image_to_base64(image)
      current_prompt['content'].append({
              "type": "image_url",
              "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}",
                  #"detail": "low"
          }
        })
    return current_prompt

  set_prompt = {
    'role': 'user',
    'content': []
  }

  # Include in-context examples if provided
  if in_context_examples:
    for example in in_context_examples:
      _append_pair(
        set_prompt, example['images'], example['prompt'])
      # interleave response
      set_prompt['content'].append({
          'type': 'text',
          'text': f"The answer should be: {example['response']}\n"
      })
    
  # add user prompt
  _append_pair(set_prompt, images, prompt)

  return set_prompt


def compose_payload(
    images: List[Union[Image.Image, np.ndarray]], 
    prompt: str, 
    system_prompt: str, 
    detail: str, 
    temperature: float, 
    max_tokens: int, 
    n: int, 
    model_name: str = "gpt-4.1", 
    return_logprobs: bool = False, 
    in_context_examples: Optional[List[Dict[str, Any]]] = None, 
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Composes a payload for the GPT API request.
    
    Args:
        images (List[Union[Image.Image, np.ndarray]]): List of images to include in the prompt.
        prompt (str): Text prompt to accompany the images.
        system_prompt (str): System prompt to guide the model's behavior.
        detail (str): Level of detail for image analysis (e.g., "auto", "low", "high").
        temperature (float): Controls randomness in output. Lower is more deterministic.
        max_tokens (int): Maximum number of tokens in the response.
        n (int): Number of completions to generate.
        model_name (str, optional): GPT model to use. Defaults to "gpt-4.1".
        return_logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
        in_context_examples (Optional[List[Dict[str, Any]]], optional): List of examples for few-shot learning. Defaults to None.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
    
    Returns:
        Dict[str, Any]: Complete payload ready for API submission.
    """
    # Prepare system message
    system_msg = {
                "role": "system",
                "content": system_prompt  # plain text, not a list
    }
    messages = [system_msg]
    # Prepare prompt message, potentially with in-context examples
    msg = prepare_prompt(
      images, prompt, in_context_examples)
    messages.append(msg)
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
        "logprobs": return_logprobs,
    }
    # reproducable output?
    if seed is not None:
      payload["seed"] = seed
    return payload


def request_gpt(
    images: Union[np.ndarray, List[Union[np.ndarray, Image.Image, str]]], 
    prompt: str, 
    system_prompt: str, 
    detail: str = "auto", 
    temp: float = 0.0, 
    n_tokens: int = 256, 
    n: int = 1, 
    return_logprobs: bool = False, 
    in_context_examples: Optional[List[Dict[str, Any]]] = None, 
    model_name: str = "gpt-4o", 
    seed: Optional[int] = None
) -> str:
    """
    Sends a request to the GPT API with images and text.
    
    Args:
        images (Union[np.ndarray, List[Union[np.ndarray, Image.Image, str]]]): 
            A single image or list of images to process.
        prompt (str): Text prompt to accompany the images.
        system_prompt (str): System prompt to guide the model's behavior.
        detail (str, optional): Level of detail for image analysis. Defaults to "auto".
        temp (float, optional): Temperature parameter. Defaults to 0.0.
        n_tokens (int, optional): Maximum number of tokens in response. Defaults to 256.
        n (int, optional): Number of completions to generate. Defaults to 1.
        return_logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
        in_context_examples (Optional[List[Dict[str, Any]]], optional): 
            List of examples for few-shot learning. Defaults to None.
        model_name (str, optional): GPT model to use. Defaults to "gpt-4o".
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
    
    Returns:
        str: The model's response text.
        
    Raises:
        ValueError: If the image type is not supported or if the API returns an error.
        AssertionError: If input image is not a valid type.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    # convert single image prompt to multiple for compatibility
    if not isinstance(images, List):
        assert isinstance(images, np.ndarray), "Provide either a numpy array, a PIL image, an image path string or a list of the above."
        images = [images]
    
    payload = compose_payload(images=images, prompt=prompt, detail=detail, system_prompt=system_prompt, n=n, temperature=temp, max_tokens=n_tokens, return_logprobs=return_logprobs, in_context_examples=in_context_examples, model_name=model_name, seed=seed)
    response = requests.post(url=API_URL, headers=headers, json=payload).json()
    
    if 'error' in response:
        raise ValueError(response['error']['message'])
    
    response = [r['message']['content'] for r in response['choices']]
    response = response[0] if n == 1 else response
    
    return response

