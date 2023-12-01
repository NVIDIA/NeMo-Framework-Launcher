# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import time

from .download_squad import download_squad

NEMO_LAUNCHER_CI = os.getenv("NEMO_LAUNCHER_CI", "False").lower() in ("true", "t", "1")


def prepare_squad_for_prompt_learning(data_dir, launcher_scripts_path):
    squad_dir = data_dir
    download_squad(squad_dir, ["v1.1"])
    squad_v1_dir = os.path.join(squad_dir, "v1.1")

    preprocess_script = launcher_scripts_path / "nemo_launcher/utils/data_utils/prompt_learning_squad_preprocessing.py"
    os.system(f"python3 {preprocess_script} " f"--data-dir={squad_v1_dir} ")


def prepare_squad_for_fine_tuning(data_dir):
    squad_dir = data_dir
    download_squad(squad_dir, ["v1.1", "xquad"])

    squad_v1_dir = os.path.join(squad_dir, "v1.1")
    squad_xquad_dir = os.path.join(squad_dir, "xquad")

    path2dev = {
        **{f"{squad_v1_dir}/train-v1.1.json": False, f"{squad_v1_dir}/dev-v1.1.json": True,},
        **{
            f"{squad_xquad_dir}/xquad.{lang}.json": True
            for lang in ["en", "es", "de", "el", "ru", "tr", "ar", "vi", "th", "zh", "hi"]
        },
    }

    for path, dev in path2dev.items():
        if (not os.path.exists(f"{os.path.splitext(path)[0]}_src.txt") or 
            not os.path.exists(f"{os.path.splitext(path)[0]}_tgt.txt") or
            not os.path.exists(f"{os.path.splitext(path)[0]}_gpt.json")
        ):
            preprocess_squad_for_fine_tuning(
                fname=path, out_fname_prefix=os.path.splitext(path)[0], dev=dev,
            )


def preprocess_squad_for_fine_tuning(fname, out_fname_prefix, dev=False):

    x = json.load(open(fname, encoding='utf8'))
    print(f"Preprocessing \"{fname}\" for fine-tuning...")
    if (os.path.exists(f'{out_fname_prefix}_src.txt') and 
        os.path.exists(f'{out_fname_prefix}_tgt.txt') and 
        os.path.exists(f'{out_fname_prefix}_gpt.json')):
        print(f"Skipped! Fine-tuning data existed at \"{out_fname_prefix}*.txt\"")
        if NEMO_LAUNCHER_CI:
            time.sleep(5)
        return
    with open(f'{out_fname_prefix}_src.txt', 'w') as f_src, open(f'{out_fname_prefix}_tgt.txt', 'w') as f_tgt, open(f'{out_fname_prefix}_gpt.json', 'w') as f_gpt:
        for i in x['data']:
            title = i['title'].replace('\n', '\\n')
            for j in i['paragraphs']:
                context = j['context'].replace('\n', '\\n')
                for k in j['qas']:
                    question = k['question'].replace('\n', '\\n')
                    if len(k['answers']) > 0:
                        if dev:
                            answer = k['answers'][0]['text'].replace('\n', '\\n')
                            f_src.write(f"Title: {title} Paragraph: {context} Question: {question}\n")
                            f_tgt.write(f"{answer}\n")
                            
                            input_text = f"{question} {title} Paragraph: {context}"
                            gpt_sample = {"input" : input_text, "output" : answer}
                            gpt_sample = json.dumps(gpt_sample)
                            f_gpt.write(f"{gpt_sample}\n")
                            
                        else:
                            for a in k['answers']:
                                answer = a['text'].replace('\n', '\\n')
                                f_src.write(f"Title: {title} Paragraph: {context} Question: {question}\n")
                                f_tgt.write(f"{answer}\n")
                                
                                input_text = f"{question} {title} Paragraph: {context}"                        
                                gpt_sample = {"input" : input_text, "output" : answer}
                                gpt_sample = json.dumps(gpt_sample)
                                f_gpt.write(f"{gpt_sample}\n")
                                                            
    print(f"Completed! Fine-tuning data saved at:")
    print(f"- \"{out_fname_prefix}_src.txt\"")
    print(f"- \"{out_fname_prefix}_tgt.txt\"")
    print(f"- \"{out_fname_prefix}_gpt.txt\"")
