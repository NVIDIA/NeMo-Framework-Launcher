# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import itertools
import json
import operator
import random
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from mpi4py import MPI
from PIL import Image
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
from sentencepiece import SentencePieceProcessor
from transformers import CLIPImageProcessor

ENGINE_DIR = Path(__file__).parent / 'plan'
VISION_ENGINE = ENGINE_DIR / 'vision_encoder.plan'
VOCAB_FILE = ENGINE_DIR / 'tokenizer.model'
TRITON_NAME = Path(__file__).parent.parent.name.upper()

DEFAULT_IMAGE_TOKEN = '<image>'  # token we get from gradio
DEFAULT_IMAGE_START_TOKEN = '<extra_id_4>'  # used in NeMo training
DEFAULT_IMAGE_END_TOKEN = '<extra_id_5>'  # used in NeMo training
DEFAULT_IMAGE_PATCH_TOKEN = '<extra_id_3>'  # doesn't matter, will get overriden by image embeddings
DEFAULT_STOP_TOKEN = '<extra_id_1>'  # used in NeMo training

STREAM_INTERVAL = 10

DTYPE = torch.bfloat16


def log(msg):
    pb_utils.Logger.log_info(f'[{TRITON_NAME}]: {msg}')


def get_single_input(request, name, default=None):
    """ Get an optional input of size [1], or return default if None """
    param = pb_utils.get_input_tensor_by_name(request, name)
    return np.squeeze(param.as_numpy()).item() if param is not None else default


def prepare_ptuning_input_ids(input_ids, im_start_token_id, vocab_size, num_img_patches):
    offset = vocab_size
    for ids in input_ids:
        start_indices = torch.where(ids == im_start_token_id)[0]
        for start_idx in start_indices:
            ids[start_idx : start_idx + num_img_patches] = torch.arange(
                offset, offset + num_img_patches, device='cuda'
            )
            offset += num_img_patches
    return input_ids


def get_image_features(image_lists, processor, vision_runner):
    """Get the image features tensor"""
    pil_images_flat = [Image.open(BytesIO(base64.b64decode(image))) for images in image_lists for image in images]
    if not pil_images_flat:
        return [None] * len(image_lists)

    processed_images_flat = processor.preprocess(pil_images_flat, return_tensors='pt')
    processed_images_flat = processed_images_flat['pixel_values']

    # FIXME: will break if we have more images than the max TRT BS
    image_features_flat = vision_runner.infer({'images': processed_images_flat})['features'].to(dtype=DTYPE)
    image_features = image_features_flat.split(list(map(len, image_lists)))
    return [feat if feat.numel() else None for feat in image_features]


def insert_image_tokens(prompt, num_img_patches, num_imgs):
    """ Insert start, patch and stop tokens in place of the placeholder image token from gradio. """
    num_img_patches = num_img_patches - 2
    replace_str = DEFAULT_IMAGE_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * num_img_patches + DEFAULT_IMAGE_END_TOKEN
    return prompt.replace(DEFAULT_IMAGE_TOKEN, replace_str, num_imgs)


def extract_by_indicies(indicies, values):
    return tuple(values[i] for i in indicies)


def get_trtllm_config():
    config_path = ENGINE_DIR / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def group_items_and_merge(inputs_dict, merge_dict):
    values = inputs_dict.values()
    merged = tuple(zip(*values))
    idx_inputs = list(enumerate(merged))
    idx_inputs.sort(key=operator.itemgetter(1))
    items = []
    for _, group in itertools.groupby(idx_inputs, key=operator.itemgetter(1)):
        _samples_idxes, _ = zip(*group)
        grouped_request = {input_name: value[_samples_idxes[0]] for input_name, value in inputs_dict.items()}
        grouped_request['indices'] = _samples_idxes
        for key, value in merge_dict.items():
            grouped_request[key] = extract_by_indicies(_samples_idxes, value)
        items.append(grouped_request)
    return items


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        config = get_trtllm_config()
        max_batch_size = config['builder_config']['max_batch_size']

        config = auto_complete_model_config.as_dict()
        if max_batch_size > 1:
            auto_complete_model_config.set_max_batch_size(max_batch_size)
            auto_complete_model_config.set_dynamic_batching()
            # TODO remove, just for debugging
            auto_complete_model_config._model_config["dynamic_batching"] = {"preferred_batch_size": max_batch_size}
        else:
            auto_complete_model_config.set_max_batch_size(0)

        return auto_complete_model_config

    def initialize(self, args):
        config = get_trtllm_config()
        world_size = config['builder_config']['tensor_parallel']
        self.hidden_size = config['builder_config']['hidden_size']
        self.max_output_len = config['builder_config']['max_output_len']
        self.vocab_size = config['builder_config']['vocab_size']

        # TRT-LLM TP needs MPI, so we spawn N processes that will listen for requests
        decoder_file = Path(__file__).parent / 'decoder.py'
        self.mpi_comm = MPI.COMM_SELF.Spawn(
            sys.executable,
            args=[str(decoder_file), f'--engine_dir={ENGINE_DIR}', f'--stream_interval={STREAM_INTERVAL}'],
            maxprocs=world_size,
        )

        self.tokenizer = SentencePieceProcessor(str(VOCAB_FILE))
        self.image_start_id = self.tokenizer.piece_to_id(DEFAULT_IMAGE_START_TOKEN)
        self.token_stop_id = self.tokenizer.eos_id()

        log(f'Loading the vision engine...')
        self.processor = CLIPImageProcessor.from_pretrained(str(ENGINE_DIR))
        self.vision_engine = EngineFromBytes(BytesFromPath(str(VISION_ENGINE)))
        self.vision_runner = TrtRunner(self.vision_engine)
        self.vision_runner.activate()
        self.num_img_patches = self.vision_runner.context.get_tensor_shape('features')[-2]
        log(f'Done')

        log(f'Loading the TRT-LLM decoders in TP={world_size} GPUs...')
        # block until all workers have loaded their decoder
        for _ in range(world_size):
            response = self.mpi_comm.recv()
            if isinstance(response, Exception):
                raise pb_utils.TritonModelException(f'Got error in decoder worker: {response}') from response
        log(f'Done')

    def execute(self, requests):
        requests_count = len(requests)
        log(f'Got {requests_count} request')
        start_time = time.perf_counter()

        # gather inputs
        prompts = [get_single_input(r, 'prompt').decode('utf-8', 'ignore') for r in requests]
        max_new_tokens = [
            max(min(int(get_single_input(r, 'max_new_tokens')), self.max_output_len), 1) for r in requests
        ]
        temperatures = [
            round(float(get_single_input(r, 'temperature', 1.0)), 2) for r in requests  # round for better grouping
        ]
        random_seeds = [get_single_input(r, 'random_seed') for r in requests]
        end_ids = [self.tokenizer.piece_to_id(get_single_input(r, 'stop').decode('utf-8', 'ignore')) for r in requests]
        image_lists = [pb_utils.get_input_tensor_by_name(r, 'images').as_numpy().flatten().tolist() for r in requests]

        # process texts
        input_ids = []
        for prompt, images in zip(prompts, image_lists):
            text = insert_image_tokens(prompt, self.num_img_patches, len(images))
            ids = torch.as_tensor(self.tokenizer.encode(text), device='cuda', dtype=torch.int32)
            input_ids.append(ids)

        # process images
        image_features = get_image_features(image_lists, self.processor, self.vision_runner)
        log(f'Tokenizer + image processing took {time.perf_counter() - start_time:.2f}s')

        # group by temperatures and seeds
        input_dict = {'temperature': temperatures, 'random_seed': random_seeds}
        merge_dict = {
            'input_ids': input_ids,
            'input_embed': image_features,
            'max_new_tokens': max_new_tokens,
            'end_id': end_ids,
        }

        items = group_items_and_merge(input_dict, merge_dict)
        lengths = [len(item['indices']) for item in items]
        print(f"Grouped {requests_count} requests into {len(items)} separate requests of sizes {lengths}")

        response_senders = [r.get_response_sender() for r in requests]
        last_response_sent = [False] * requests_count
        last_tokens = [
            0
        ] * requests_count  # Keeping track of number of currently generated tokens. We can't reset it in for loop because of early stoping.

        num_new_tokens = 0
        time_to_first_token_first_group = 0
        time_to_first_token_last_group = 0
        current_requests = 0
        generation_start_time = time.perf_counter()

        for item_id, item in enumerate(items):
            group_input_ids = prepare_ptuning_input_ids(
                item['input_ids'], self.image_start_id, self.vocab_size, self.num_img_patches
            )
            item['input_ids'] = group_input_ids
            group_indices = item.pop('indices')
            group_max_new_tokens = item.pop('max_new_tokens')  # Max_new_tokens are supported on parent side
            group_end_ids = item.pop('end_id')  # Users end_ids are supported on parent side
            item[
                'end_id'
            ] = self.token_stop_id  # We always pass end_id used in training to avoid long garbage outputs generation
            item['max_new_tokens'] = max(group_max_new_tokens)  # We always take max of the group
            if item['random_seed'] is None:
                item['random_seed'] = random.randrange(2 ** 32)
            self.mpi_comm.bcast(item, root=MPI.ROOT)

            while True:
                response = self.mpi_comm.recv(source=0)
                if isinstance(response, Exception):
                    raise pb_utils.TritonModelException(f'Got error in decoder worker: {response}') from response

                group_response_senders = extract_by_indicies(group_indices, response_senders)
                if response is None:
                    current_requests += len(group_indices)
                    group_last_response_sent = extract_by_indicies(group_indices, last_response_sent)
                    for rs, finished in zip(group_response_senders, group_last_response_sent):
                        if not finished:
                            rs.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

                    # Do we also need timing per group? Client withing different groups will need to wait for others to finish.
                    if current_requests == requests_count:
                        time_spent = time.perf_counter() - start_time
                        time_spent_on_generation = time.perf_counter() - generation_start_time

                        msg = f"""
##### SENT FINAL RESPONSE #####
              Num request in batch:    {requests_count} | Grouped as: {lengths}
                        Total Time:    {time_spent:.2f}s
              Time Generation Time:    {time_spent_on_generation:.2f}s
                  Time per request:    {time_spent/requests_count:.2}s
Time to first tokens (first group):    {time_to_first_token_first_group:.2f}s
 Time to first tokens (last group):    {time_to_first_token_last_group:.2f}s
                             Tok/s:    {num_new_tokens / time_spent:.2f} tok/s
           Tok/s [only generation]:    {num_new_tokens / time_spent_on_generation:.2f} tok/s
###############################
"""
                        log(msg)
                    break
                else:
                    for i, rs, in_ids, out_ids, user_stop_id, user_max_tokens in zip(
                        group_indices,
                        group_response_senders,
                        group_input_ids,
                        response,
                        group_end_ids,
                        group_max_new_tokens,
                    ):
                        if not last_response_sent[i]:
                            if (
                                user_stop_id in out_ids
                                or self.token_stop_id in out_ids
                                or len(out_ids) > user_max_tokens
                            ):
                                if user_stop_id in out_ids:  # User passed different stop_id
                                    end_id = out_ids.index(user_stop_id)
                                    out_ids = out_ids[:end_id]
                                if (
                                    self.token_stop_id in out_ids
                                ):  # Decoder is padding shorter responses, no need to process that
                                    end_id = out_ids.index(self.token_stop_id)
                                    out_ids = out_ids[:end_id]
                                if (
                                    len(out_ids) > user_max_tokens
                                ):  # Decoder generated more max_tokens than user wanted
                                    out_ids = out_ids[:user_max_tokens]
                                last_response_sent[i] = True

                            if (
                                time_to_first_token_first_group == 0
                            ):  # Get time spent until first generated tokens for first client in first group
                                time_to_first_token_first_group = time.perf_counter() - start_time

                            if (
                                item_id == len(items) - 1 and time_to_first_token_last_group == 0
                            ):  # Get time spent until first generated tokens for first client in last group
                                time_to_first_token_last_group = time.perf_counter() - start_time

                            num_new_tokens += len(out_ids) - last_tokens[i]
                            last_tokens[i] = len(out_ids)

                            out_text = self.tokenizer.decode(out_ids)
                            response = pb_utils.InferenceResponse(
                                output_tensors=[pb_utils.Tensor('generated_text', np.array([out_text], np.object_))]
                            )
                            rs.send(response)
                            if last_response_sent[i]:
                                rs.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                                print(f"Sent final response to ID [{i}]")

    def finalize(self):
        self.mpi_comm.bcast(None, root=MPI.ROOT)
        log('Finalized python backend')
