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

import argparse
import json
import logging
from pathlib import Path

import tensorrt_llm
import torch
from mpi4py import MPI
from tensorrt_llm.runtime import ModelConfig, SamplingConfig


def get_decoder(engine_dir, rank, world_size):
    config_path = engine_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_config = ModelConfig(
        num_heads=config['builder_config']['num_heads'] // world_size,
        num_kv_heads=config['builder_config']['num_heads'] // world_size,
        hidden_size=config['builder_config']['hidden_size'] // world_size,
        vocab_size=config['builder_config']['vocab_size'],
        num_layers=config['builder_config']['num_layers'],
        gpt_attention_plugin=config['plugin_config']['gpt_attention_plugin'],
        remove_input_padding=config['plugin_config']['remove_input_padding'],
        use_prompt_tuning=config['builder_config']['use_prompt_tuning'],
    )
    assert model_config.use_prompt_tuning
    assert model_config.remove_input_padding

    precision = config['builder_config']['precision']

    serialize_path = engine_dir / f'ammo_{precision}_tp{world_size}_rank{rank}.engine'
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()

    runtime_mapping = tensorrt_llm.Mapping(world_size=world_size, rank=rank)

    decoder = tensorrt_llm.runtime.GenerationSession(model_config, engine_buffer, runtime_mapping)

    return decoder, config['builder_config']


def throttle_generator(generator, stream_interval):
    """ To not spam responses, we only yield every N TRT-LLM output. """
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield i, out
    if i % stream_interval:
        yield i, out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_dir', type=Path)
    parser.add_argument('--stream_interval', type=int)
    args = parser.parse_args()

    comm = MPI.Comm.Get_parent()
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    logging.basicConfig(level=logging.DEBUG, format=f'[worker {rank}] [%(levelname)s] %(message)s', force=True)

    try:
        torch.cuda.set_device(rank)
        decoder, config = get_decoder(args.engine_dir, rank, world_size)

        # no embedding offset needed, so task and ptuning vocab size can be zero
        tasks = torch.zeros((config['max_batch_size']), device='cuda', dtype=torch.int32)
        prompt_vocab_size = torch.zeros(1, device='cuda', dtype=torch.int32)
        padded_input_embed = torch.empty(
            config['max_input_len'] * config['max_batch_size'],
            config['hidden_size'],
            device='cuda',
            dtype=decoder._tensor_dtype('prompt_embedding_table'),
        )

        logging.info('Sending ready signal')
        comm.send(True, dest=0)
        while True:
            # wait and receive item from master
            logging.debug('Waiting for master item')
            item = comm.bcast(None, root=0)
            logging.debug('Got master item')
            if item is None:
                logging.debug('Exiting')
                break

            end_id = item['end_id']
            temperature = item['temperature']
            random_seed = item['random_seed']
            max_new_tokens = min(item['max_new_tokens'], config['max_output_len'])
            input_ids = item['input_ids']
            input_embed = item['input_embed']

            if min(ids.shape[0] for ids in input_ids) > config['max_input_len']:
                if rank == 0:
                    logging.debug('Exceeded max length')
                    comm.send(None, dest=0)
                continue

            top_k = int(temperature < 1e-4)
            context_size = max(ids.shape[0] for ids in input_ids)
            batch_size = len(input_ids)
            sampling_config = SamplingConfig(
                end_id=end_id, pad_id=end_id, temperature=temperature, top_k=top_k, top_p=1.0
            )
            sampling_config.random_seed = random_seed

            decoder.setup(batch_size, context_size, max_new_tokens)
            input_embed = [embed.flatten(0, 1) for embed in input_embed if embed is not None]
            if input_embed:
                cat_embed = torch.cat(input_embed)
                padded_input_embed[: cat_embed.shape[0]] = cat_embed

            # run the generation
            tensors = [torch.flatten(t) for t in input_ids]
            data = torch.concat(tensors).unsqueeze(0)
            row_lengths = [t.size(0) for t in tensors]
            row_lengths = torch.tensor(row_lengths, dtype=torch.int32, device='cuda')

            output_ids = decoder.decode(
                data,
                row_lengths,
                sampling_config,
                prompt_embedding_table=padded_input_embed,
                tasks=tasks[:batch_size],
                prompt_vocab_size=prompt_vocab_size,
            )

            if rank == 0:
                output_ids = output_ids[:, 0, :].tolist()
                for idx in range(batch_size):
                    input_size = input_ids[idx].shape[0]
                    output_ids[idx] = output_ids[idx][input_size:]
                comm.send(output_ids, dest=0)

            if rank == 0:
                logging.info('Sending final None')
                comm.send(None, dest=0)

    except Exception as e:
        logging.exception('Exception in decoder')
        comm.send(e, dest=0)
