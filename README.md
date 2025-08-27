# Kwai Keye-VL


<div align="center">
  <img src="asset/keye_logo_2.png" width="100%" alt="Kwai Keye-VL Logo">
</div>

<!-- <font size=7> -->
<div align='center'>
[🍎 Home Page](https://kwai-keye.github.io/) 
[📖 Technical Report](https://arxiv.org/abs/2507.01949) 
[📊 Models](https://huggingface.co/Kwai-Keye/Keye-VL-1.5-8B) 
[🚀 Demo](https://huggingface.co/spaces/Kwai-Keye/Keye-VL-8B-Preview)
</div>
<!-- </font> -->

## 🔥 News
* **`2025.08.28`** 🌟 We are excited to introduce **Keye-VL-1.5**, a more powerful version! By incorporating innovative `Slow-Fast Video Encoding strategy`, `new LongCoT Cold-Start data pipeline`, and `advanced RL training strategies`, Keye-VL-1.5 reaches new heights in video understanding, image comprehension, and reasoning capabilities. Plus, it now supports an extended context length of up to **128k** tokens for handling longer conversations and complex tasks. Stay tuned for more groundbreaking innovations! 
* **`2025.07.08`** 🌟 Keye-VL is supported by [swift](https://github.com/modelscope/ms-swift) and [vLLM](https://github.com/vllm-project/vllm). Feel free to use it without hesitation!
* **`2025.07.03`** 🌟 We are excited to announce the release of our comprehensive technical report!  You can read it now at [arxiv](https://arxiv.org/abs/2507.01949).  
* **`2025.06.26`** 🌟 We are very proud to launch **Kwai Keye-VL**, a cutting-edge multimodal large language model meticulously crafted by the **Kwai Keye Team** at [Kuaishou](https://www.kuaishou.com/). As a cornerstone AI product within Kuaishou's advanced technology ecosystem, Keye excels in video understanding, visual perception, and reasoning tasks, setting new benchmarks in performance. Our team is working tirelessly to push the boundaries of what's possible, so stay tuned for more exciting updates!



<div align="center">
  <img src="asset/teaser.png" width="100%" alt="Kwai Keye-VL Performance">
</div>

## Contents <!-- omit in toc -->

- [🔥 News](#-news)
- [📐 Quick Start](#-quick-start)
  - [Preprocess and Inference](#preprocess-and-inference)
  - [Evaluation](#evaluation)
- [👀 Architecture and Training Strategy](#-architecture-and-training-strategy)
  - [🌟 Pre-Train](#-pre-train)
  - [🌟 Post-Train](#-post-train)
- [📈 Experimental Results](#-experimental-results)
- [✒️ Citation](#️-citation)

## 📐 Quick Start
### Preprocess and Inference

See [keye-vl-utils/README.md](keye-vl-utils/README.md) for details. ```Keye-vl-utils``` contains a set of helper functions for processing and integrating visual language information with Keye Series Model.

#### Install

```bash
pip install keye-vl-utils
```

#### Keye-VL Inference

```python
from transformers import AutoModel, AutoProcessor
from keye_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model_path = "Kwai-Keye/Keye-VL-8B-Preview"

model = AutoModel.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2", trust_remote_code=True,
).to('cuda')

# You can set the maximum tokens for a video through the environment variable VIDEO_MAX_PIXELS
# based on the maximum tokens that the model can accept. 
# export VIDEO_MAX_PIXELS = 32000 * 28 * 28 * 0.9


# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
messages = [
    # Image
    ## Local file path
    [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    ## Image URL
    [{"role": "user", "content": [{"type": "image", "image": "http://path/to/your/image.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    ## Base64 encoded image
    [{"role": "user", "content": [{"type": "image", "image": "data:image;base64,/9j/..."}, {"type": "text", "text": "Describe this image."}]}],
    ## PIL.Image.Image
    [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": "Describe this image."}]}],
    ## Model dynamically adjusts image size, specify dimensions if required.
    [{"role": "user", "content": [{"type": "image", "image": "file:///path/to/your/image.jpg", "resized_height": 280, "resized_width": 420}, {"type": "text", "text": "Describe this image."}]}],
    # Video
    ## Local video path
    [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4"}, {"type": "text", "text": "Describe this video."}]}],
    ## Local video frames
    [{"role": "user", "content": [{"type": "video", "video": ["file:///path/to/extracted_frame1.jpg", "file:///path/to/extracted_frame2.jpg", "file:///path/to/extracted_frame3.jpg"],}, {"type": "text", "text": "Describe this video."},],}],
    ## Model dynamically adjusts video nframes, video height and width. specify args if required.
    [{"role": "user", "content": [{"type": "video", "video": "file:///path/to/video1.mp4", "fps": 2.0, "resized_height": 280, "resized_width": 280}, {"type": "text", "text": "Describe this video."}]}],
]

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype="auto", device_map="auto").to('cuda')
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs).to("cuda")
print(inputs)
generated_ids = model.generate(**inputs)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

#### Deployment
- We recommend using vLLM for fast Keye-VL-8B-Preview deployment and inference.

##### Install
```bash
pip install keye-vl-utils "vllm>=0.9.2"
```

##### Offline Inference
```bash
# refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#inference-locally

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from keye_vl_utils import process_vision_info

model_path = "/hetu_group/jky/playground_hhd_2/2025/20250626_keye/Keye-VL-8B-Preview"

llm = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 10, "video": 10},
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=0.3,
    max_tokens=256,
)

# image
image_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://s1-11508.kwimgs.com/kos/nlav11508/mllm_all/ziran_jiafeimao_11.jpg",
            },
            {"type": "text", "text": "Describe this image./think"},
        ],
    },
]

# video
video_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "http://s2-11508.kwimgs.com/kos/nlav11508/MLLM/videos_caption/98312843263.mp4",
            },
            {"type": "text", "text": "Describe this video./think"},
        ],
    },
]

# Here we use video messages as a demonstration
messages = video_messages

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages, return_video_kwargs=True
)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
if video_inputs is not None:
    mm_data["video"] = video_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
    # FPS will be returned in video_kwargs
    "mm_processor_kwargs": video_kwargs,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)
```

##### Online Serving
- Serve
```bash
vllm serve \
    Kwai-Keye/Keye-VL-8B-Preview \
    --tensor-parallel-size 8 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

- Openai Chat Completion Client
```python
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from keye_vl_utils import process_vision_info
import requests


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# image url
image_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://s1-11508.kwimgs.com/kos/nlav11508/mllm_all/ziran_jiafeimao_11.jpg"
                },
            },
            {"type": "text", "text": "Describe this image./think"},
        ],
    },
]

chat_response = client.chat.completions.create(
    model="Kwai-Keye/Keye-VL-8B-Preview",
    messages=image_messages,
)
print("Chat response:", chat_response)

# image base64-encoded

import base64

image_path = "/path/to/local/image.png"
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
encoded_image_text = encoded_image.decode("utf-8")
image_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image;base64,{encoded_image_text}"
                },
            },
            {"type": "text", "text": "Describe this image./think"},
        ],
    },
]

chat_response = client.chat.completions.create(
    model="Kwai-Keye/Keye-VL-8B-Preview",
    messages=image_messages,
)
print("Chat response:", chat_response)

# video, refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#start-an-openai-api-service
video_messages = [
    {"role": "user", "content": [
        {"type": "video", "video": "http://s2-11508.kwimgs.com/kos/nlav11508/MLLM/videos_caption/98312843263.mp4"},
        {"type": "text", "text": "Describe this video./think"}]
    },
]

def prepare_message_for_vllm(content_messages):
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}


video_messages, video_kwargs = prepare_message_for_vllm(video_messages)


chat_response = client.chat.completions.create(
    model="Kwai-Keye/Keye-VL-8B-Preview",
    messages=video_messages,
    max_tokens=128,
    extra_body={
        "mm_processor_kwargs": video_kwargs
    }
)

print("Chat response:", chat_response)
```

### Evaluation
See [evaluation/KC-MMBench/README.md](evaluation/KC-MMBench/README.md) for details.

## 👀 Architecture and Training Strategy

<div align="center">
  <img src="asset/architecture.png" width="100%" alt="Kwai Keye Architecture">
  <i> The Kwai Keye-VL model architecture is based on the Qwen3-8B language model and incorporates a vision encoder initialized from the open-source SigLIP. It supports native dynamic resolution, preserving the original aspect ratio of images by dividing each into a 14x14 patch sequence. A simple MLP layer then maps and merges the visual tokens. The model uses 3D RoPE for unified processing of text, image, and video information, establishing a one-to-one correspondence between position encoding and absolute time to ensure precise perception of temporal changes in video information.</i>
</div>


### 🌟 Pre-Train

<div align="center">
  <img src="asset/pre-train.png" width="100%" alt="Kwai Keye Pretraining">
  <i>The Kwai Keye pre-training pipeline, featuring a four-stage progressive strategy: Image-Text Matching, ViT-LLM Alignment, Multi-task Pre-training, and Annealing with model merging.</i>
</div>
<details>
  <summary>More Details</summary>

  #### Pre-training Data: Massive, High-Quality, Diverse

  - **Diversity**: Includes image-text pairs, videos, pure text, etc., with tasks such as fine-grained description, OCR, Q&A, localization, and more.
  - **High Quality**: Data is filtered using CLIP scores and VLM discriminators, and MinHASH is used for deduplication to prevent data leakage.
  - **Self-Built Datasets**: High-quality internal datasets are specifically constructed, especially for detailed captions and Chinese OCR, to compensate for the shortcomings of open-source data.

  #### Training Process: Four-Stage Progressive Optimization
  Kwai Keye-VL adopts a four-stage progressive training strategy:

  - **Stage 0 (Visual Pre-training)**: Continuously pre-trains the visual encoder to adapt to internal data distribution and support dynamic resolution.
  - **Stage 1 (Cross-Modal Alignment)**: Freezes the backbone model and trains only the MLP to establish robust image-text alignment at low cost.
  - **Stage 2 (Multi-Task Pre-training)**: Unlocks all parameters to comprehensively enhance the model's visual understanding capabilities.
  - **Stage 3 (Annealing Training)**: Fine-tunes with high-quality data to further improve the model's fine-grained understanding capabilities.

  Finally, Kwai Keye-VL explores isomorphic heterogeneous fusion technology by averaging parameters of annealed training models with different data ratios, reducing model bias while retaining multidimensional capabilities, thereby enhancing the model's robustness.

</details>


### 🌟 Post-Train

The post-training phase of Kwai Keye-VL-1.5 is meticulously designed into two phases, aiming to comprehensively enhance the model's performance, especially its reasoning ability in complex tasks. This is a key breakthrough for achieving advanced cognitive functions.

#### Stage I. No-Reasoning Training: SFT+MPO


<details>
  <summary>More Details</summary>

- **Stage I.1: Supervised Fine-Tuning (SFT)**
  - Data Composition: Includes 5 million multimodal data, built on a diverse task classification system (70,000 tasks) using the self-developed TaskGalaxy framework. High-difficulty data is selected by multimodal large models and manually annotated to ensure data quality and challenge.

- **Stage I.2: Mixed Preference Optimization (MPO)**
  - Data Composition: Comprises open-source data and pure text preference data. Bad cases from the SFT model are used as quality prompts, and preference data is generated through rejection sampling using Qwen2.5VL 72B and SFT models, with manual scoring and ranking.

</details>

<div align="center">
  <img src="asset/post1.png" width="100%" alt="Kwai Keye Post-Training">
  <i>Keye-VL-1.5's post-training is composed of Non-Reasoning Stage and Reasoning Stage, non-reasoning stage includes large-scale SFT and MPO, Reasoning Stage includes three steps: CoT Cold-Start, General RL and Alignment RL.</i>
</div>

#### Stage II. Reasoning Training: Core Breakthrough for Complex Cognition

<div align="center">
  <img src="asset/post2.png" width="100%" alt="LongCoT data generation pipeline">
  <br>
  <i>Overview of our five-step automated LongCoT data generation pipeline. .</i>
</div>


<details>
  <summary>More Details</summary>

- **Step II.1: CoT Cold-Start**
  - Objective: Cold-start the model's chain of thought reasoning ability, allowing it to mimic human step-by-step thinking.
  - Data Composition: Combines non-reasoning data (330,000), reasoning data (230,000), auto-reasoning data (20,000), and agentic reasoning data (100,000) to teach the model different modes.
    - Thinking Data: Focuses on high-difficulty perception and reasoning scenarios like math, science, charts, complex Chinese, and OCR, using multimodal large models for multiple sampling and evaluation to build over 70,000 complex thought chain data.
    - Pure Text Data: Constructs a pure text long thought chain dataset from dimensions like code, math, science, instruction following, and general reasoning tasks.
    - Auto-Think Data: Automatically selects "think" or "no_think" modes based on the complexity of prompts, enabling adaptive reasoning mode switching.
    - Think with Image Data: 100,000 agent data entries, asking Qwen 2.5 VL-72B if image operations (e.g., cropping, rotating, enhancing contrast) are needed to simplify problems or improve answer quality, combined with external sandbox code execution to empower the model to solve problems by writing code to manipulate images or perform mathematical calculations.
  - Training Strategy: Trains with a mix of four modes to achieve cold-start in different reasoning modes.
- **Step II.2: CoT-Mix RL**
  - Objective: Deeply optimize the model's comprehensive abilities in multimodal perception, reasoning, pure text math, short video understanding, and agentic tasks through reinforcement learning based on the chain of thought, making the reasoning process more robust and efficient.
  - Data Composition: Covers complex tasks from multimodal perception (complex text recognition, object counting), multimodal reasoning, high-difficulty math problems, short video content understanding to Think with Image.
  - Training Strategy: Uses a mix-mode GRPO algorithm for reinforcement learning, where reward signals evaluate both the correctness of results and the consistency of the process and results, ensuring synchronized optimization of reasoning processes and final outcomes.
- **Step II.2: Iterative Alignment**
  - Objective: Address common issues like repetitive crashes and poor logic in model-generated content, and enable spontaneous reasoning mode selection to enhance final performance and stability.
  - Data Composition: Constructs preference data through Rejection Fine-Tuning (RFT), combining rule-based scoring (judging repetition, instruction following, etc.) and model scoring (cognitive scores provided by large models) to rank various model responses, building a high-quality preference dataset.
  - Training Strategy: Multi-round iterative optimization with the constructed "good/bad" preference data pairs through the MPO algorithm. This aims to correct model generation flaws and ultimately enable it to intelligently and adaptively choose whether to activate deep reasoning modes based on problem complexity.

</details>

## 📈 Experimental Results

<div align="center">
  <img src="asset/performance.png" width="100%" alt="Kwai Keye-VL Performance">
</div>

1. Keye-VL-8B establishes itself with powerful, state-of-the-art perceptual abilities that are competitive with leading models. 
2. Keye-VL-8B demonstrates exceptional proficiency in video understanding. Across a comprehensive suite of authoritative public video benchmarks, including Video-MME, Video-MMMU, TempCompass, LongVideoBench, and MMVU, the model's performance significantly surpasses that of other top-tier models of a comparable size.
3. In evaluation sets that require complex logical reasoning and mathematical problem-solving, such as WeMath, MathVerse, and LogicVista, Kwai Keye-VL-8B displays a strong performance curve. This highlights its advanced capacity for logical deduction and solving complex quantitative problems.


## ✒️ Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
@misc{kwaikeyeteam2025kwaikeyevltechnicalreport,
      title={Kwai Keye-VL Technical Report}, 
      author={Kwai Keye Team},
      year={2025},
      eprint={2507.01949},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.01949}, 
}
```

## Acknowledgement

Kwai Keye-VL is developed based on the codebases of the following projects: [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384), [Qwen3](https://github.com/QwenLM/Qwen3), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We sincerely thank these projects for their outstanding work.
