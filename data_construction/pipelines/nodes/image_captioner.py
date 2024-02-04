import torch
from typing import *
import importlib
import tempfile
import copy

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torchvision.transforms import functional as TF
from .base import Node


def minicpmv_process_prompt(model, tokenizer, messages: List[Dict[str, str]]):
    prompt = ''
    for i, msg in enumerate(messages):
        role = msg['role']
        content = msg['content']
        assert role in ['user', 'assistant']
        if i == 0:
            assert role == 'user', 'The role of first msg should be user'
            content = tokenizer.im_start + tokenizer.unk_token * model.config.query_num + tokenizer.im_end + '\n' + content
        prompt += '<用户>' if role=='user' else '<AI>'
        prompt += content
    prompt += '<AI>'
    return prompt


def load_minicpmv_model_tokenizer(model_path: str = 'openbmb/MiniCPM-V', dtype: torch.dtype = torch.float16, device: Union[str, torch.device] = 'cuda'):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def minicpmv_predict_caption(model, tokenizer, images: torch.Tensor, *, max_new_length: int = 76):
    prompt = minicpmv_process_prompt(model, tokenizer,[{'role': 'user', 'content': 'Briefly caption this image, inlcuding the style, subjects and actions.'}])
    generation_config = {
        'top_p': 0.8,
        'top_k': 100,
        'temperature':0.6,
        'do_sample': True
    }
    images_pil = [TF.to_pil_image(img) for img in images.unbind()]
    num_images = len(images_pil)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        responses, vision_hidden_states = model.generate(
            data_list=[prompt] * num_images,
            max_inp_length=2048,
            img_list=[[img] for img in images_pil],
            tokenizer=tokenizer,
            max_new_tokens=max_new_length,
            vision_hidden_states=None,
            return_vision_hidden_states=True,
            **generation_config
        )
    return responses


def load_qwen_model_tokenizer(model_path: str = "Qwen/Qwen-VL-Chat-Int4", device: Union[str, torch.device] = 'cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True).eval()
    return model, tokenizer


def qwen_predict_caption(model, tokenizer, images: torch.Tensor, *, max_new_length: int = 76):
    qwen_generation_utils = importlib.import_module(importlib.import_module(model.__module__).__package__).qwen_generation_utils
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = max_new_length
    stop_words_ids = qwen_generation_utils.get_stop_words_ids(generation_config.chat_format, tokenizer)

    # Save images to temporary files
    image_files = []
    for i, img in enumerate(images.unbind(0)):
        file = tempfile.NamedTemporaryFile(suffix='.png', delete=True)
        TF.to_pil_image(img).save(file)
        image_files.append(file)

    # Encode 
    batch_raw_text, batch_context_tokens = [], []
    for i in range(len(image_files)):
        query = tokenizer.from_list_format([
            {
                'image': image_files[i].name,
                'text': 'Briefly caption this image.'
            }
        ])
        raw_text, context_tokens = qwen_generation_utils.make_context(
            tokenizer,
            query,
            history=None,
            system='You are a helpful assistant.',
            max_window_size=generation_config.max_window_size,
            chat_format=generation_config.chat_format,
        )
        batch_raw_text.append(raw_text)
        batch_context_tokens.append(context_tokens)

    # Generate
    input_ids = torch.tensor(batch_context_tokens).to(model.device)
    outputs = model.generate(
        input_ids,
        stop_words_ids=stop_words_ids,
        return_dict_in_generate=False,
        generation_config=generation_config,
    )

    # Decode
    batch_response = [qwen_generation_utils.decode_tokens(
        output,
        tokenizer,
        raw_text_len=len(batch_raw_text[i]),
        context_length=input_ids.shape[1],
        chat_format=model.generation_config.chat_format,
        verbose=False,
        errors='replace'
    ) for i, output in enumerate(outputs.unbind(0))]

    # Remove temporary files
    for file in image_files:
        file.close()

    return batch_response


class MiniCPMVCaptioner(Node):
    def __init__(self, image_key: str = "image", caption_key: str = "caption", batch_size: int = 32, device: Union[str, torch.device] = 'cuda'):
        super().__init__()
        self.image_key = image_key
        self.caption_key = caption_key
        self.batch_size = batch_size
        self.device = device
    
    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        """
        Predict depth from image using ZoeDepth.

        Args:
            image: (N, 3, H, W) tensor of images.
        Returns:
            depth: (N, H, W) tensor of metric depth.
        """
        n_images = data[self.image_key].shape[0]
        model, tokenizer = self.get_lazy_component(
            'mini-cpm-v', 
            load_minicpmv_model_tokenizer, 
            init_kwargs={'device': self.device}, 
            pipe=pipe
        )
        captions = []
        for i in range(0, n_images, self.batch_size):
            captions.extend(minicpmv_predict_caption(model, tokenizer, data[self.image_key][i:i + self.batch_size]))
        data[self.caption_key] = captions
        return data


class QwenVLCaptioner(Node):
    def __init__(self, image_key: str = "image", caption_key: str = "caption", batch_size: int = 32):
        raise NotImplementedError("QwenVLCaptioner has problem with batch inference and decodine.")
        super().__init__()
        self.image_key = image_key
        self.caption_key = caption_key
        self.batch_size = batch_size
    
    def __call__(self, data: Dict[str, torch.Tensor], pipe=None):
        """
        Predict depth from image using ZoeDepth.

        Args:
            image: (N, 3, H, W) tensor of images.
        Returns:
            depth: (N, H, W) tensor of metric depth.
        """
        n_images = data[self.image_key].shape[0]
        model, tokenizer = self.get_lazy_component('qwen-vl', load_qwen_model_tokenizer, pipe=pipe)
        captions = []
        for i in range(0, n_images, self.batch_size):
            captions.extend(qwen_predict_caption(model, tokenizer, data[self.image_key][i:i + self.batch_size]))
        data[self.caption_key] = captions
        return data