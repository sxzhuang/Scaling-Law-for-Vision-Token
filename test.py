# import base64
# from openai import OpenAI

# client = OpenAI(api_key="EMPTY", base_url="http://10.74.197.104:50017/v1")

# img_path = "build_dataset_from_ebook/pride_dataset_square/images/pride_prejudice_double_C2B1_C23B2.jpg"
# with open(img_path, "rb") as f:
#     b64 = base64.b64encode(f.read()).decode("utf-8")

# DEFAULT_PROMPT = "You are a high-fidelity OCR (Optical Character Recognition) to Markdown converter. Your goal is to transcribe the content of the provided image into Markdown format with pixel-perfect accuracy regarding text and structure. Convert the visible text and structural elements in the image into raw Markdown code. # Output Format (CRITICAL) \n You must output the final result wrapped inside a Markdown code block. Your response must look exactly like this structure:```markdown \n (The transcribed content goes here) \n```"

# resp = client.chat.completions.create(
#     # model="deepseek-ai/deepseek-vl2",
#     model="deepseek-ai/deepseek-vl-1.3b-chat",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": DEFAULT_PROMPT},
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
#             ],
#         }
#     ],
#     max_tokens=3800,
# )
# choice = resp.choices[0] if resp.choices else None
# finish_reason = choice.finish_reason if choice else None
# print(choice.message.content if choice else "")
# print(f"finish_reason={finish_reason} (truncated={finish_reason == 'length'})")

from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://10.74.197.104:50016/v1")

img_path = Path("build_dataset_from_ebook/pride_dataset_square/images/pride_prejudice_double_C2B1_C31B1.jpg")
image_url = img_path.resolve().as_uri()  # file:///abs/path/to/xxx.jpg

DEFAULT_PROMPT = "You are a high-fidelity OCR (Optical Character Recognition) to Markdown converter. Your goal is to transcribe the content of the provided image into Markdown format with pixel-perfect accuracy regarding text and structure. Convert the visible text and structural elements in the image into raw Markdown code. # Output Format (CRITICAL) \n You must output the final result wrapped inside a Markdown code block. Your response must look exactly like this structure:```markdown \n (The transcribed content goes here) \n```"

resp = client.chat.completions.create(
    # model="deepseek-ai/deepseek-vl-1.3b-chat",
    model="OpenGVLab/InternVL3_5-8B",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DEFAULT_PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ],
    max_tokens=3800,
)
choice = resp.choices[0] if resp.choices else None
finish_reason = choice.finish_reason if choice else None
print(choice.message.content if choice else "")
print(f"finish_reason={finish_reason} (truncated={finish_reason == 'length'})")