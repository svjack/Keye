# Evaluation Datasets 
This project is based on the excellent [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) framework.

Based on the in-house short video data, we constructed 6 datasets for **Keye** and other Vision-Language Models (VLMs) like **Qwen2.5-VL** and **InternVL** to evaluate performance.

## Tasks
| Task           | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| CPV            | The task of predicting product attributes in e-commerce.                    |
| Hot_Videos_Aggregation    | The task of determining whether multiple videos belong to the same topic.   |
| Collection_Order     | The task of determining the logical order between multiple videos with the same topic. |
| Pornographic_Comment    | The task of whether short video comments contain pornographic content.      |
| High_Like      | A binary classification task to determine the rate of likes of a short video. |
| SPU            | The task of determining whether two items are the same product in e-commerce. |

These datasets can be downloaded from [Hugging Face (HF)](https://huggingface.co/datasets/Kwai-Keye/KC-MMbench). 

## Performance 
| Task           | Qwen2.5-VL-3B | Qwen2.5-VL-7B | InternVL-3-8B | MiMo-VL-7B | Kwai Keye-VL-8B |
| -------------- | ------------- | ------------- | ------------- | ------- | ---- |
| CPV            | 12.39         | 20.08         | 14.95         | 16.66   | 55.13 |
| Hot_Videos_Aggregation    | 42.38         | 46.35         | 52.31         | 49.00   | 54.30 |
| Collection_Order    | 36.88         | 59.83         | 64.75         | 78.68   | 84.43 |
| Pornographic_Comment    | 56.61         | 56.08         | 57.14         | 68.25   | 71.96 |
| High_Like      | 48.85         | 47.94         | 47.03         | 51.14   | 55.25 |
| SPU            | 74.09         | 81.34         | 75.64         | 81.86   | 87.05 |

## Example of Evaluation

### Config
Here is an example of an evaluation using VLMs on our datasets. The following configuration needs to be added to the config file.

```python
{

    "model":'...'
    "data": {
        "CPV": {
            "class": "KwaiVQADataset",
            "dataset": "CPV"
        },
        "Hot_Videos_Aggregation": {
            "class": "KwaiVQADataset",
            "dataset": "Hot_Videos_Aggregation"
        },
        "Collection_Order": {
            "class": "KwaiVQADataset",
            "dataset": "Collection_Order"
        },
        "Pornographic_Comment": {
            "class": "KwaiYORNDataset",
            "dataset": "Pornographic_Comment"
        },
        "High_like":{
            "class":"KwaiYORNDataset",
            "dataset":"High_like"
        },
        "SPU": {
            "class": "KwaiYORNDataset",
            "dataset": "SPU"
        }
    }
}
```

### Evaluation of Keye-VL-8B-Preview
```bash
cd evaluation/KC-MMBench/scripts/Keye-VL-8B-Preview

# download dataset
git clone https://huggingface.co/datasets/KwaiKeye/KC-MMbench
cp -r ./KC-MMbench/images/* ./KC-MMbench/subsets/
export LMUData=./KC-MMbench/subsets

# run eval
bash run.sh
```
