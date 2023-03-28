# Multimodal Dataset Preparation

## Overview

We provide utilities to download and prepare datasets used for training multimodal (e.g. image and text) models. Datasets are downloaded from a [Hugging Face data repository](https://huggingface.co/datasets), and prepared into the [WebDataset](https://github.com/webdataset/webdataset) format for efficient training. More specifically, data preparation consists of the following sub-stages:

- `download_parquet`: Parquet files consisting of text (captions) and image URLs are downloaded from a Hugging Face repository.
- `download_images`: Images are downloaded from the URLs, and are packed with the captions into tar files, following the WebDataset format.
- `reorganize_tar`: (Optional) Due to a variety of reason (e.g. unstable network, images are taken down), some images will always fail to be downloaded, resulting in uneven tar files with different number of examples each. If you are using a training sampler that does not support uneven tar files, you need to re-organize the contents of the tar files so that each one contains exactly the same number of image-text pairs. 
- `precache_encodings`: (Optional) If you are training a model with frozen encoders (e.g. Stable Diffusion, eDiff-i), you can precache (precompute) image and/or text encodings (embeddings) in this sub_stage to improve training throughput. 
- `generate_wdinfo`: (Optional) Generate the `wdinfo.pkl` file, which stores information on dataset shards.

## Configuration for Precaching
Configuration for precaching can become long and complex for certain models, so a separate yaml file is used for clarity.

### General Format
Precached encodings are saved in the format of WebDataset. Each tar file contains one pickle file to store all the modality embeddings for each training example, optionally along with the original image or text files

```text
t0_r0_0.tar
|---- 00000.pickle
|---- 00000.jpg (optional)
|---- 00000.txt (optional)
|---- 00001.pickle
|---- 00001.jpg (optional)
|---- 00001.txt (optional)
...
```

Each pickle file stores one python dictionary object, with key value pairs storing the embedding name and the embeddings.

### Config
Configuration for the above format is specified in `encodings` in the precaching yaml file (e.g. mulimodal/precache_sd.yaml).

`encodings` specifies a list of embeddings to be saved in the pickle file. Each entry can have the following attributes:
- `modality`: either image or text
- `extension`: file extension to use in the tar file
- `key`: dictionary key for the encoding. It is recommended to follow the format `{model_name}-{model_variant}_{modality}` if applicable. e.g. `clip-vit-large-patch14_text`
- `precision`: precision of the stored tensors
- `store_pad_tokens`: Whether to store the PAD tokens. Not storing PAD tokens can significantly reduce disk usage, but the training script must account for this. Ignored for image modality.
- `encoder_config`: This dictionary must contain `cls` which points to the location of the encoder class. The rest of the parameters are treated as kwargs to initiate the encoder class.
  - Note: the encoder class must implement an `encode` or `__call__` function. If `store_pad_tokens`, this function must return the encoded tensor. Else, this function must return a tuple of (encoded_tensor, text_mask).

Note that it is not required to have only one encoding per modality, if there are multiple encoders. The `encodings` config is designed as a list to account for this. For example, it's possible to have one image embedding from CLIP, one text embedding from CLIP, and a second text embedding from T5. 