from typing import Iterable, List
import torch
import numpy as np
from transformers import AutoTokenizer, LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
from llava.conversation import simple_conv_multimodal

MAX_TOKENS = 64
PROMPT = simple_conv_multimodal.get_prompt() + "Human: "

def load_llava(params_path):
    # Load model with the newer LlavaForConditionalGeneration
    processor = LlavaProcessor.from_pretrained(params_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        params_path, torch_dtype=torch.float16
    ).cuda()
    
    # Get the tokenizer from the processor
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    @torch.inference_mode()
    def inference_fn(
        images: Iterable[Image.Image], queries: Iterable[Iterable[str]]
    ) -> List[List[str]]:
        assert len(images) == len(queries)
        assert np.all(len(queries[0]) == len(q) for q in queries)

        queries = np.array(queries)  # (batch_size, num_queries_per_image)
        
        # The newer API handles image token insertion automatically
        # We'll process each batch of queries for each image
        results = []

        for i, image in enumerate(images):
            image_queries = queries[i]
            batch_outputs = []
            
            for query in image_queries:
                # Prepare the prompt with the query
                full_prompt = PROMPT + query
                
                # Process inputs - the processor handles image tokens automatically
                inputs = processor(text=full_prompt, images=image, return_tensors="pt")
                
                # Move inputs to GPU
                inputs = {k: v.to("cuda", dtype=torch.float16 if k == "pixel_values" else None) 
                         for k, v in inputs.items()}
                
                # Generate response
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=False
                )
                
                # Decode the generated tokens
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Clean up the output
                if "Human:" in generated_text and "Assistant:" in generated_text:
                    generated_text = generated_text.split("Assistant:", 1)[1].strip()
                
                batch_outputs.append(generated_text)
            
            results.append(batch_outputs)

        return results

    return inference_fn
