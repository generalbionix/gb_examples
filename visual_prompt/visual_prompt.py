"""
Visual prompting module for interacting with GPT vision models.
This module provides the VisualPrompter base class and VisualPrompterGrounding implementation
for creating visual prompts with images, sending them to GPT models, and parsing responses
for visual grounding and object identification tasks.
"""

import os
import pickle
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Any, Optional, Tuple, Set
import supervision as sv

from visual_prompt.llm_utils import request_gpt
from visual_prompt.utils import load_config
from visual_prompt.utils import (
    create_subplot_image,
    mask2box,
)
from visual_prompt.postprocessing import (
    masks_to_marks,
    refine_marks,
)
from visual_prompt.visualizer import load_mark_visualizer


class VisualPrompter:

    def __init__(
        self,
        prompt_root_dir: str,
        system_prompt_name: str,
        config: Dict[str, Any],
        prompt_template: str,
        inctx_examples_name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """
        Base class for sending visual prompts to GPT.
        Initializes the VisualPrompter with a path to the system prompt file,
        a configuration dictionary for the GPT request, and a prompt template.

        Args:
            prompt_root_dir (str): Path to the directory containing the prompts.
            system_prompt_name (str): Name of the .txt file containing the system prompt.
            config (Dict[str, Any]): A dictionary containing the arguments for the GPT request
                                     except for 'images', 'prompt', and 'system_prompt'.
            prompt_template (str): An f-string template for constructing the user prompt.
            inctx_examples_name (Optional[str]): Path to a pickle binary file containing in-context examples.
                                        Defaults to None (zero-shot).
            debug (bool): Whether to print GPT responses.
        """
        self.prompt_root_dir = prompt_root_dir
        self.system_prompt_path = os.path.join(prompt_root_dir,
                                               system_prompt_name)
        self.request_config = config
        self.prompt_template = prompt_template
        self.system_prompt = self._load_text_prompt(self.system_prompt_path)
        self.debug = debug

        self.do_inctx = False
        if inctx_examples_name is not None:
            self.do_inctx = True
            self.inctx_examples = pickle.load(
                open(os.path.join(self.prompt_root_dir, inctx_examples_name),
                     "rb"))

    @staticmethod
    def _load_text_prompt(prompt_path: str) -> str:
        """
        Reads the text prompt from a specified .txt file.

        Args:
            prompt_path (str): Path to the .txt file containing the prompt.

        Returns:
            str: The content of the text prompt file.
            
        Raises:
            ValueError: If the text prompt file is not found.
        """
        try:
            with open(prompt_path, "r") as file:
                text_prompt = file.read().strip()
            return text_prompt
        except FileNotFoundError:
            raise ValueError(f"Text prompt file not found: {prompt_path}")

    def prepare_image_prompt(self, image: Union[Image.Image, np.ndarray, str],
                             data: Dict[str, Any]) -> Tuple[List[Union[Image.Image, np.ndarray]], Dict[str, Any]]:
        """
        Placeholder method for preparing the image inputs.
        This will be implemented in subclasses.

        Args:
            image (Union[Image.Image, np.ndarray, str]):
                Image (PIL, numpy or path string) to construct the visual prompt from.
            data (Dict[str, Any]): Additional data that are useful for `prepare_image_prompt` method.
                
        Returns:
            Tuple[List[Union[Image.Image, np.ndarray]], Dict[str, Any]]: 
                A tuple containing the processed images and related metadata.
                
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def parse_response(self, response: str, data: Dict[str, Any]) -> Any:
        """
        Placeholder method for parsing the response from GPT.
        This will be implemented in subclasses.

        Args:
            response (str): The response from GPT.
            data (Dict[str, Any]): Additional data that are useful for `prepare_image_prompt` method.

        Returns:
            Any: Parsed response data (to be defined by subclasses).
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def request(
        self,
        image: Union[Image.Image, np.ndarray, str],
        data: Dict[str, Any],
        text_query: Optional[str] = None,
    ) -> Dict[int, Any]:
        """
        Sends the constructed prompt to GPT via the OpenAI API.

        Args:
            image (Union[Image.Image, np.ndarray, str]):
                Image (PIL, numpy or path string) to construct the visual prompt from.
            data (Dict[str, Any]): Additional data that are useful for `prepare_image_prompt` method.
            text_query (Optional[str]): The text query that will be inserted into the prompt template.

        Returns:
            Dict[int, Any]: The parsed response from GPT (processed by subclasses).
        """
        # Construct the prompt using the provided template and user input
        if text_query is not None:
            text_prompt = self.prompt_template.format(user_input=text_query)
        else:
            text_prompt = self.prompt_template  # no text query

        # Prepare images based on markers
        image_prompt, image_prompt_utils = self.prepare_image_prompt(
            image, data)

        # Extract relevant settings from the config dictionary
        temperature: float = self.request_config.get("temperature", 0.0)
        max_tokens: int = self.request_config.get("n_tokens", 256)
        n: int = self.request_config.get("n", 1)
        model_name: str = self.request_config.get("model_name", "gpt-4.1")

        # Call the request_gpt function to get the response
        response: str = request_gpt(
            images=image_prompt,
            prompt=text_prompt,
            system_prompt=self.system_prompt,
            temp=temperature,
            n_tokens=max_tokens,
            n=n,
            in_context_examples=self.inctx_examples if self.do_inctx else None,
            model_name=model_name,
        )
        if self.debug:
            print("\033[92mGPT response:\033[0m")
            print("\033[92m" + response.strip() + "\033[0m")
            print()

        # Parse and return the response (this will be subclassed to define behavior)
        return self.parse_response(response, image_prompt_utils)


class VisualPrompterGrounding(VisualPrompter):

    def __init__(self, config_path: str, debug: bool = False) -> None:
        """
        Initializes the VisualPrompterGrounding class with a YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
            debug (bool, optional): Whether to print GPT responses. Defaults to False.
        """
        # Load config from YAML file
        cfg = load_config(config_path)
        self.image_size = (cfg.image_size_h, cfg.image_size_w)
        self.image_crop = cfg.image_crop
        self.cfg = cfg.grounding
        self.use_subplot_prompt = self.cfg.use_subplot_prompt

        # Extract config related to VisualPrompter and initialize superclass
        config_for_prompter = self.cfg.request
        config_for_visualizer = self.cfg.visualizer

        # Initialize superclass
        super().__init__(
            prompt_root_dir=cfg.prompt_root_dir,
            system_prompt_name=self.cfg.prompt_name,
            config=config_for_prompter,
            prompt_template=self.cfg.prompt_template,
            inctx_examples_name=self.cfg.inctx_prompt_name
            if self.cfg.do_inctx else None,
            debug=debug,
        )

        # Create visualizer using the visualizer config in YAML
        self.visualizer = load_mark_visualizer(config_for_visualizer)

    def prepare_image_prompt(
        self, image: Union[Image.Image, np.ndarray],
        data: Dict[str, np.ndarray]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Prepares the image prompt by resizing and overlaying segmentation masks.

        Args:
            image (Union[Image.Image, np.ndarray]): The input image (as a PIL image or numpy array).
            data (Dict[str, np.ndarray]): 
                Contains `masks`, boolean array of size (N, H, W) for N instance segmentation masks.
                (Optional) Contains `labels`, list of label IDs to name the markers.
                
        Returns:
            List[np.ndarray]: The processed image or a list containing both the raw and marked images if configured.
            Dict[str, Any]: The detection markers, potentially refined.
        """
        masks = data["masks"]
        labels = data['labels'] if ('labels' in data.keys()
                                    and data['labels'] is not None) else list(
                                        range(1,
                                              len(masks) + 1))

        image_size_h = self.image_size[0]
        image_size_w = self.image_size[1]
        image_crop = self.image_crop
        include_raw_image = self.cfg.include_raw_image
        use_subplot_prompt = self.use_subplot_prompt

        # Resize image and masks if sizes differ
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
            image = np.array(image_pil)

        if image_pil.size != (image_size_w, image_size_h):
            image_pil = image_pil.resize((image_size_w, image_size_h),
                                         Image.Resampling.LANCZOS)
            masks = np.array([
                np.array(
                    Image.fromarray(mask).resize((image_size_w, image_size_h),
                                                 Image.LANCZOS)).astype(bool)
                for mask in masks
            ])
            image = np.array(image_pil)

        if image_crop:
            image = image[image_crop[0]:image_crop[2],
                          image_crop[1]:image_crop[3]].copy()
            masks = np.stack([
                m[image_crop[0]:image_crop[2],
                  image_crop[1]:image_crop[3]].copy() for m in masks
            ])

        # Process markers from masks
        markers = masks_to_marks(masks, labels=labels)

        # Optionally refine markers
        if self.cfg.do_refine_marks:
            refine_kwargs = self.cfg.refine_marks
            markers = refine_marks(markers, **refine_kwargs)

        if use_subplot_prompt:
            # Use separate legend image
            assert (
                include_raw_image is True
            ), "`use_subplot_prompt` should be set to True together with `include_raw_image`"
            # Masked cropped object images
            boxes = [mask2box(mask) for mask in masks]
            crops = []
            for mask, box in zip(masks, boxes):
                masked_image = image.copy()
                masked_image[mask == False] = 127
                crop = masked_image[box[1]:box[3], box[0]:box[2]]
                crops.append(crop)
            subplot_size = self.cfg.subplot_size
            marked_image = create_subplot_image(crops,
                                                h=subplot_size,
                                                w=subplot_size)

        else:
            # Use the visualizer to overlay the markers on the image
            marked_image = self.visualizer.visualize(
                image=np.array(image).copy(), marks=markers)

        # Prepare the image prompt
        img_prompt = [marked_image]
        if include_raw_image:
            img_prompt = [image.copy(), marked_image]
        output_data = {
            "markers": markers,
            "raw_image": image.copy(),
            'labels': labels,
        }

        return img_prompt, output_data

    def parse_response(
        self, response: str, data: Dict[str, Any]
    ) -> Tuple[Dict[int, sv.Detections], np.ndarray, List[int]]:
        """
        Parses the GPT response to extract relevant mask IDs and returns corresponding markers.

        Args:
            response (str): The raw response from GPT, which contains the IDs of the objects identified.
            data (Dict[str, Any]): Contains `markers`, a dictionary where keys are mask IDs and values are corresponding mask data.
                                   Contains `labels`, list of label IDs to name the markers.

        Returns:
            Tuple[Dict[int, sv.Detections], np.ndarray, List[int]]: A tuple containing:
                - Dictionary of selected markers based on GPT's response
                - The combined output mask
                - List of output IDs
                
        Raises:
            Exception: If parsing the response fails.
        """
        markers = data["markers"]
        labels = list(data['labels'])
        try:
            # Extract the portion of the response that contains the final answer IDs
            output_IDs_str = (response.split("final answer is:")[1].replace(
                ".", "").strip())
            output_IDs = eval(output_IDs_str)  # Convert string to list of IDs

            # Convert to `labels` indexing
            output_IDs_ret = [labels.index(x) for x in output_IDs]

            # Return the masks corresponding to the extracted IDs
            outputs = {mark: markers[mark] for mark in output_IDs_ret}
            output_mask = np.zeros_like(markers[0].mask.squeeze(0))
            for _, mark in outputs.items():
                output_mask[mark.mask.squeeze(0) == True] = True

            return outputs, output_mask, output_IDs

        except Exception as e:
            print(f"Failed parsing response: {e}")
            return {}, np.array([]), []

