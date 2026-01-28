import os
import re
from tqdm import tqdm
import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import config
import argparse
parser = argparse.ArgumentParser(description="Run DeepSeek OCR batch job.")
parser.add_argument("--base_size", type=int, help="Override BASE_SIZE from config.")
parser.add_argument("--image_size", type=int, help="Override IMAGE_SIZE from config.")
parser.add_argument("--input_path", type=str, help="Directory containing input images.")
parser.add_argument("--output_path", type=str, help="Directory to store OCR predictions.")
parser.add_argument("--batch_size", type=int, default=500, help="Number of images processed per batch.")
args = parser.parse_args()
config.BASE_SIZE = args.base_size
config.IMAGE_SIZE = args.image_size
config.INPUT_PATH = args.input_path
config.OUTPUT_PATH = args.output_path
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, MAX_CONCURRENCY, CROP_MODE, NUM_WORKERS

from concurrent.futures import ThreadPoolExecutor
import glob
from PIL import Image
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True, 
    max_model_len=8192,
    swap_space=0,
    max_num_seqs = MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=40, window_size=90, whitelist_token_ids= {128821, 128822})] #window for fast；whitelist_token_ids: <td>,</td>

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m' 

def clean_formula(text):

    formula_pattern = r'\\\[(.*?)\\\]'
    
    def process_formula(match):
        formula = match.group(1)

        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        
        formula = formula.strip()
        
        return r'\[' + formula + r'\]'

    cleaned_text = re.sub(formula_pattern, process_formula, text)
    
    return cleaned_text

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    # mathes_image = []
    mathes_other = []
    for a_match in matches:
        mathes_other.append(a_match[0])
    return matches, mathes_other

def process_single_image(image):
    """single image"""
    prompt_in = prompt
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images = [image], bos=True, eos=True, cropping=CROP_MODE)},
    }
    return cache_item


if __name__ == "__main__":

    # INPUT_PATH = OmniDocBench images path

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # print('image processing until processing prompts.....')

    print(f'{Colors.RED}glob images.....{Colors.RESET}')

    images_path = sorted(glob.glob(f'{INPUT_PATH}/*'))
    if not images_path:
        raise FileNotFoundError(f"No images found under {INPUT_PATH}")
    batch_size = max(1, args.batch_size)

    prompt = PROMPT

    output_path = OUTPUT_PATH
    os.makedirs(output_path, exist_ok=True)

    def chunked(seq, size):
        for start in range(0, len(seq), size):
            yield seq[start:start + size]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        progress_bar = tqdm(total=len(images_path), desc="Pre-processed images")
        for batch_idx, batch_paths in enumerate(chunked(images_path, batch_size), start=1):
            images = []
            for image_path in batch_paths:
                with Image.open(image_path) as image:
                    images.append(image.convert('RGB'))

            batch_inputs = list(
                executor.map(process_single_image, images)
            )
            progress_bar.update(len(batch_paths))

            outputs_list = llm.generate(
                batch_inputs,
                sampling_params=sampling_params
            )

            for output, image in zip(outputs_list, batch_paths):
                content = output.outputs[0].text
                mmd_det_path = output_path + image.split('/')[-1].replace('.jpg', '_det.md')

                with open(mmd_det_path, 'w', encoding='utf-8') as afile:
                    afile.write(content)

                content = clean_formula(content)
                matches_ref, mathes_other = re_match(content)
                for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
                    content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')

                mmd_path = output_path + image.split('/')[-1].replace('.jpg', '.md')

                with open(mmd_path, 'w', encoding='utf-8') as afile:
                    afile.write(content)

            del images
            del batch_inputs
        progress_bar.close()
