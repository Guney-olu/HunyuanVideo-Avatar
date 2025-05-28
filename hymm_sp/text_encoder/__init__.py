from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict, Any
from copy import deepcopy

import torch, os
import torch.nn as nn
from transformers import (
    CLIPTextModel, CLIPTokenizer, LlavaForConditionalGeneration,
    LlamaTokenizerFast, AutoProcessor # Added AutoProcessor
)
from transformers.utils import ModelOutput
from ..constants import TEXT_ENCODER_PATH, TOKENIZER_PATH, PRECISION_TO_TYPE

CPU_OFFLOAD = int(os.environ.get("CPU_OFFLOAD", 0))
# Using a global logger instance is generally not ideal in libraries,
# but following the existing pattern.
from loguru import logger as global_logger # Assuming loguru is used globally
print(f'text_encoder: cpu_offload={CPU_OFFLOAD}')


def use_default(value, default):
    return value if value is not None else default

def load_text_encoder(text_encoder_type,
                      text_encoder_precision=None,
                      text_encoder_path=None,
                      logger=None, # Parameter for local logger
                      device=None
                      ):
    effective_logger = logger if logger is not None else global_logger
    if text_encoder_path is None:
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
    effective_logger.info(f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}")

    target_dtype = PRECISION_TO_TYPE.get(text_encoder_precision, torch.float16) # Default to float16 if None

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=target_dtype)
        # text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm # Already part of CLIPTextModel
    elif text_encoder_type == "llava-llama-3-8b":
        text_encoder = LlavaForConditionalGeneration.from_pretrained(
            text_encoder_path,
            low_cpu_mem_usage=True, # Good for large models
            torch_dtype=target_dtype
        )
        # Accessing final_layer_norm, ensure path is correct for your LLaVA model version
        if hasattr(text_encoder, 'language_model') and hasattr(text_encoder.language_model, 'model') and hasattr(text_encoder.language_model.model, 'norm'):
            text_encoder.final_layer_norm = text_encoder.language_model.model.norm
        else:
            effective_logger.warning(f"Could not set final_layer_norm for LLaVA model {text_encoder_type}. Check model structure.")
            text_encoder.final_layer_norm = None # Or some default nn.Identity()
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

    text_encoder.requires_grad_(False)
    text_encoder.eval() # Set to eval mode

    effective_logger.info(f"Text encoder loaded with dtype: {text_encoder.dtype}")

    if device is not None:
        text_encoder = text_encoder.to(device)
        effective_logger.info(f"Text encoder moved to device: {text_encoder.device}")


    return text_encoder, text_encoder_path

def load_processor_or_tokenizer( # Renamed to reflect it can load a processor
                   processor_tokenizer_type, # Renamed
                   processor_tokenizer_path=None, # Renamed
                   padding_side="right",
                   logger=None # Parameter for local logger
                   ):
    effective_logger = logger if logger is not None else global_logger
    if processor_tokenizer_path is None:
        # Assuming TOKENIZER_PATH for LLaVA points to the same model hub ID or local path
        processor_tokenizer_path = TOKENIZER_PATH[processor_tokenizer_type]

    effective_logger.info(f"Loading processor/tokenizer ({processor_tokenizer_type}) from: {processor_tokenizer_path}")

    if processor_tokenizer_type == "clipL":
        obj = CLIPTokenizer.from_pretrained(processor_tokenizer_path, max_length=77)
    elif processor_tokenizer_type == "llava-llama-3-8b":
        try:
            # AutoProcessor will load the correct LlavaProcessor or LlavaNextProcessor
            obj = AutoProcessor.from_pretrained(processor_tokenizer_path, padding_side=padding_side)
            effective_logger.info(f"Loaded LLaVA processor of type: {type(obj)}")
        except Exception as e:
            effective_logger.error(f"Failed to load AutoProcessor for LLaVA from {processor_tokenizer_path}: {e}. "
                               f"Ensure the path contains processor_config.json.")
            raise
    else:
        raise ValueError(f"Unsupported processor/tokenizer type: {processor_tokenizer_type}")

    return obj, processor_tokenizer_path


@dataclass
class TextEncoderModelOutput(ModelOutput):
    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None


class TextEncoder(nn.Module):
    def __init__(self,
                 text_encoder_type: str,
                 max_length: int,
                 text_encoder_precision: Optional[str] = None,
                 text_encoder_path: Optional[str] = None,
                 tokenizer_type: Optional[str] = None, # Will be processor type for LLaVA
                 tokenizer_path: Optional[str] = None, # Will be processor path for LLaVA
                 output_key: Optional[str] = None,
                 use_attention_mask: bool = True,
                 input_max_length: Optional[int] = None,
                 prompt_template_video: Optional[dict] = None,
                 hidden_state_skip_layer: Optional[int] = None,
                 apply_final_norm: bool = False,
                 reproduce: bool = False,
                 logger=None, # Instance logger
                 device=None, # Target device
                 ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision # Stored but dtype primarily set during model load
        self.model_path = text_encoder_path
        
        self.processor_tokenizer_type = tokenizer_type if tokenizer_type is not None else text_encoder_type
        self.processor_tokenizer_path_arg = tokenizer_path # Store user-provided path
        
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert use_attention_mask is True, "Attention mask is True required when training videos."
        self.input_max_length = input_max_length if input_max_length is not None else max_length
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.logger = logger if logger is not None else global_logger
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.use_video_template = self.prompt_template_video is not None
        # ... (template validation logic - seems fine) ...

        if "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output" # CLIP uses pooler_output or last_hidden_state
        elif "llava" in text_encoder_type or "llama" in text_encoder_type: # Adjusted for LLaVA
            self.output_key = output_key or "last_hidden_state" # LLaMA part of LLaVA uses last_hidden_state
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.model_path_loaded = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=self.device # Pass target device
        )
        self.dtype = self.model.dtype # Get actual dtype from loaded model

        # For LLaVA, this loads the LlavaProcessor. For CLIP, CLIPTokenizer.
        _processor_tokenizer_path = self.processor_tokenizer_path_arg
        if _processor_tokenizer_path is None: # If user didn't specify, use same as model path for LLaVA
             _processor_tokenizer_path = self.model_path_loaded if "llava" in self.processor_tokenizer_type else self.processor_tokenizer_path_arg

        self.processor_or_tokenizer, self.processor_or_tokenizer_path_loaded = load_processor_or_tokenizer(
            processor_tokenizer_type=self.processor_tokenizer_type,
            processor_tokenizer_path=_processor_tokenizer_path,
            padding_side="right", # LLaMA typically uses right padding for tokenizer
            logger=self.logger
        )
        # Ensure processor's device matches model's device if it has components to move (though processors are usually CPU)

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path_loaded})"

    @staticmethod
    def apply_text_to_template(text, template):
        # ... (seems fine) ...
        if isinstance(template, str):
            return template.format(text)
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def _prepare_text_with_template_and_llava_suffix(self, text, data_type='video', name='person'):
        # Apply video template if configured
        if self.use_video_template:
            if data_type == 'video':
                prompt_template = self.prompt_template_video["template"]
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            if isinstance(text, (list, tuple)):
                processed_text = [self.apply_text_to_template(one_text, prompt_template) for one_text in text]
            elif isinstance(text, str):
                processed_text = self.apply_text_to_template(text, prompt_template)
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")
        else:
            processed_text = text

        # Apply LLaVA specific suffix if this is a LLaVA encoder
        if self.text_encoder_type == "llava-llama-3-8b":
            llava_suffix = f'\nThe {name} looks like<image>' # Or just "<image>" if that's what your model expects
            # Standard LLaVA often uses "<image>\n{prompt}" or similar, where processor handles <image>
            # Your suffix seems to be custom. The LLaVA processor needs to know about <image> as a special token.
            if isinstance(processed_text, list):
                # Ensure each item is a string before concatenation
                final_text = [str(t) + llava_suffix for t in processed_text]
            elif isinstance(processed_text, str):
                final_text = str(processed_text) + llava_suffix
            else: # Should not happen if use_video_template output is string or list of strings
                raise TypeError(f"Processed text has unexpected type: {type(processed_text)}")
        else:
            final_text = processed_text
        
        return final_text

    def get_processed_inputs(self, text: Union[str, List[str]], images: Optional[torch.Tensor] = None,
                             data_type: str = 'video', name: str = 'person'):
        """
        Process raw text and optional images using the appropriate tokenizer or processor.
        For LLaVA, this uses the LLaVA processor to handle both text and images.
        For CLIP, this tokenizes text.
        """
        final_text = self._prepare_text_with_template_and_llava_suffix(text, data_type, name)

        common_kwargs = dict(return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)

        if self.text_encoder_type == "llava-llama-3-8b":
            # LLaVA processor handles text and images.
            # `images` should be a BCHW tensor or list of PIL images.
            # The processor will create 'input_ids', 'attention_mask', and 'pixel_values'.
            if images is not None:
                inputs = self.processor_or_tokenizer(text=final_text, images=images, **common_kwargs)
            else:
                # LLaVA processor can also process text-only prompts.
                inputs = self.processor_or_tokenizer(text=final_text, **common_kwargs)
        else: # For CLIP or other text-only encoders
            inputs = self.processor_or_tokenizer(final_text, return_attention_mask=True, **common_kwargs)
            # CLIP tokenizer doesn't inherently handle 'images' argument in this way.
            if images is not None and self.logger:
                self.logger.warning(f"Images provided to a non-LLaVA encoder ({self.text_encoder_type}), they will be ignored by tokenizer.")
        
        return inputs

    def encode(self, text_or_processed_inputs: Union[str, List[str], Dict[str, torch.Tensor]],
               images: Optional[torch.Tensor] = None, # For LLaVA, if text_or_processed_inputs is raw text
               use_attention_mask: Optional[bool] = None,
               output_hidden_states: Optional[bool] = False,
               # do_sample: Optional[bool] = None, # Not typically used for getting embeddings
               hidden_state_skip_layer: Optional[int] = None,
               # return_texts: Optional[bool] = False, # Not typically used for getting embeddings
               data_type: str = 'video', # For template and LLaVA suffix
               name_for_template: str = 'person' # For LLaVA suffix
              ):
        
        effective_use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        effective_hidden_state_skip_layer = use_default(hidden_state_skip_layer, self.hidden_state_skip_layer)
        # effective_do_sample = use_default(do_sample, not self.reproduce) # Usually False for embeddings

        if isinstance(text_or_processed_inputs, (str, list)):
            # If raw text, process it with images if provided (for LLaVA)
            processed_inputs = self.get_processed_inputs(text_or_processed_inputs, images=images,
                                                         data_type=data_type, name=name_for_template)
        elif isinstance(text_or_processed_inputs, dict):
            processed_inputs = text_or_processed_inputs
            # If inputs are pre-processed, ensure 'pixel_values' is present for LLaVA if an image was involved.
            # And if raw 'images' are also passed now, it's ambiguous. Prioritize pre-processed.
            if images is not None and "pixel_values" not in processed_inputs and self.text_encoder_type == "llava-llama-3-8b":
                if self.logger: self.logger.warning("Received pre-tokenized inputs and raw images for LLaVA. Raw images might be ignored if not used to create 'pixel_values' in pre-tokenized input.")
        else:
            raise TypeError("Input must be str, list of str, or Dict of tensors.")

        if CPU_OFFLOAD:
            self.model.to(self.device) # Move to target device (e.g. cuda)
            if self.logger: self.logger.debug(f'encode prompt: move text_encoder to {self.device}')

        model_call_kwargs = {
            "input_ids": processed_inputs["input_ids"].to(self.device),
            "output_hidden_states": output_hidden_states or (effective_hidden_state_skip_layer is not None)
        }

        if effective_use_attention_mask and "attention_mask" in processed_inputs:
            model_call_kwargs["attention_mask"] = processed_inputs["attention_mask"].to(self.device)
        
        # CRITICAL FOR LLAVA: Pass 'pixel_values' if the processor added them
        if "pixel_values" in processed_inputs:
            # Ensure dtype matches the model's expected input dtype for images
            # LLaVA vision tower might expect float16/bfloat16 if model is in that precision
            image_input_dtype = self.model.vision_tower.dtype if hasattr(self.model, 'vision_tower') else self.dtype
            model_call_kwargs["pixel_values"] = processed_inputs["pixel_values"].to(dtype=image_input_dtype, device=self.device)

        outputs = self.model(**model_call_kwargs)

        # Extracting the correct hidden state
        if effective_hidden_state_skip_layer is not None:
            # This logic assumes 'outputs.hidden_states' exists and is a list/tuple
            # For LLaVA, it might be outputs.language_model_outputs.hidden_states
            if self.text_encoder_type == "llava-llama-3-8b" and hasattr(outputs, 'language_model_outputs'):
                all_hidden_states = outputs.language_model_outputs.hidden_states
            else:
                all_hidden_states = outputs.hidden_states
            
            last_hidden_state = all_hidden_states[-(effective_hidden_state_skip_layer + 1)]
            
            if effective_hidden_state_skip_layer > 0 and self.apply_final_norm and self.model.final_layer_norm is not None:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            if self.text_encoder_type == "llava-llama-3-8b":
                # Standard LLaVA output has language_model_outputs
                if hasattr(outputs, 'language_model_outputs') and outputs.language_model_outputs is not None:
                    last_hidden_state = outputs.language_model_outputs.last_hidden_state
                elif hasattr(outputs, 'last_hidden_state'): # Fallback if structure is flatter
                     last_hidden_state = outputs.last_hidden_state
                else: # Should not happen with standard LlavaForConditionalGeneration
                    raise AttributeError("Could not extract last_hidden_state from LLaVA model output.")
            elif "clip" in self.text_encoder_type:
                if self.output_key == "pooler_output":
                    last_hidden_state = outputs.pooler_output
                else: # "last_hidden_state"
                    last_hidden_state = outputs.last_hidden_state
            else: # Fallback for other models if self.output_key is set
                last_hidden_state = outputs[self.output_key]
        
        # Cropping logic (seems specific to video templates)
        final_attention_mask_for_output = model_call_kwargs.get("attention_mask", None)
        if self.use_video_template:
            if data_type == 'video':
                crop_start = self.prompt_template_video.get("crop_start", -1)
            else:
                # This path was problematic before, ensure it's handled or error out
                raise ValueError(f"Video template cropping logic encountered unsupported data_type: {data_type}")
            
            if crop_start > 0:
                last_hidden_state = last_hidden_state[:, crop_start:]
                if final_attention_mask_for_output is not None:
                    final_attention_mask_for_output = final_attention_mask_for_output[:, crop_start:]
        
        if CPU_OFFLOAD:
            self.model.to('cpu')
            torch.cuda.empty_cache()
            if self.logger: self.logger.debug(f'encode prompt successful: move text_encoder to cpu')
        
        # Determine hidden_states_list for output
        hidden_states_list_for_output = None
        if output_hidden_states or (effective_hidden_state_skip_layer is not None):
            if self.text_encoder_type == "llava-llama-3-8b" and hasattr(outputs, 'language_model_outputs'):
                hidden_states_list_for_output = outputs.language_model_outputs.hidden_states
            else:
                hidden_states_list_for_output = getattr(outputs, 'hidden_states', None)

        return TextEncoderModelOutput(last_hidden_state, final_attention_mask_for_output, hidden_states_list_for_output)

    def forward(self, text: Union[str, List[str]], 
                images: Optional[torch.Tensor] = None, # For LLaVA
                use_attention_mask: Optional[bool] = None, 
                output_hidden_states: Optional[bool] = False, 
                # do_sample: Optional[bool] = False, # Not used for embeddings
                hidden_state_skip_layer: Optional[int] = None, 
                # return_texts: Optional[bool] = False, # Not used for embeddings
                data_type: str = 'video', 
                name_for_template: str = 'person'):
        
        return self.encode(text_or_processed_inputs=text,
                           images=images,
                           use_attention_mask=use_attention_mask,
                           output_hidden_states=output_hidden_states,
                           hidden_state_skip_layer=hidden_state_skip_layer,
                           data_type=data_type,
                           name_for_template=name_for_template)