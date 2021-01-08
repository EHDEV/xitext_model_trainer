import os
import shutil
import torch
from psutil import cpu_count
from pathlib import Path
from onnxruntime import InferenceSession, SessionOptions, get_all_providers
from onnxruntime_tools import optimizer
from transformers.convert_graph_to_onnx import convert
from scipy.special import softmax

from transformers import DistilBertTokenizerFast


class TorchToONNX:

    def __init__(
            self,
            torch_model_path,
            onnx_model_dir,
            tokenizer,
            model_type='bert',
            onnx_model_filename='model.onnx',
            onnx_model_optimized_filename='model_optimized.onnx'
    ):
        self.torch_model = torch.load(torch_model_path)
        self.onnx_model_local_dir = onnx_model_dir
        self.tokenizer = tokenizer
        self.model_type=model_type,
        self.onnx_model_filename = onnx_model_filename
        self.onnx_model_optimized_filename = onnx_model_optimized_filename

        self.onnx_model_full_path = Path(self.onnx_model_local_dir / self.onnx_model_filename)
        self.optimized_model_path = Path(self.onnx_model_local_dir / self.onnx_model_optimized_filename)

    def convert_torch_to_onnx(self):
        """
         Converts torch model into ONNX + optimizes it into a small onnx file
           - requires an empty directory
           - currently model config is limited to bert (768 hidden size x 12 attentions)
        :return:
        """
        if os.path.exists(self.onnx_model_local_dir):
            shutil.rmtree(self.onnx_model_local_dir)

        convert(
            framework="pt",
            model=self.torch_model,
            output=self.onnx_model_full_path,
            opset=11,
            tokenizer=self.tokenizer
        )

        # logger.debug(f'torch model successfully converted to onnx {self.onnx_model_full_path}')

        # # Mixed precision conversion for bert-base-cased model converted from Pytorch
        optimized_model = optimizer.optimize_model(
            str(self.onnx_model_full_path),
            model_type=self.model_type,
            num_heads=12,
            hidden_size=768
        )

        optimized_model.convert_model_float32_to_float16()
        optimized_model.save_model_to_file(str(self.optimized_model_path))

    def create_model_for_provider(self, provider: str = 'CPUExecutionProvider') -> InferenceSession:
        """
        prepares optimized model for inference
        :param provider:
        :return:
        """
        assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

        # Few properties than might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1

        # Load the model as a graph and prepare the CPU backend
        return InferenceSession(str(self.optimized_model_path), options, providers=[provider])

    def run_inference(self, text, tokenizer, proba=True):
        model_inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        logits = self.create_model_for_provider().run(None, inputs_onnx)
        if not proba:
            return logits[0]

        return softmax(logits[0], axis=1)


"""
    Below code demonstrates how this is to be used
"""

'''
from pathlib import Path 
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

text = 'This is some medical device text that needs to be manufactured in china'

tmp = Path('/content/drive/MyDrive/elite/elitelearn/models/medical-device-timeline-classifier/pytorch_model.bin')
omp = Path('/content/drive/MyDrive/elite/elitelearn/models/medical-device-timeline-classifier/onnx')

tto = TorchToONNX(
    torch_model_path=tmp,
    onnx_model_dir=omp,
    tokenizer=tokenizer
)

tto.convert_torch_to_onnx()
'''

'''
How to run inference

from scipy.special import softmax

sentence = ["""The endurant II bifurcated stent grafts are being offered in 31 configurations based on 
various combinations of the proximal main body diameters, covered lengths of the bifurcated stent graft, 
and the distal diameter of the ipsilateral leg. We are introducing a shorter ipsilateral leg and 
fixed covered length of 103 mm limits the number of endurant IIs configuration to five; 
which will aid the physician with inventory management"""]


model_inputs = tokenizer(
    sentence,
    padding=True,
    add_special_tokens=True,
    max_length=64,
    return_tensors='pt'
)

inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

cpum = tto.create_model_for_provider("CPUExecutionProvider")
# model_inputs = tokenizer.encode_plus(["click to see what's in store"], return_tensors="pt", return_token_type_ids=False)
res = cpum.run(None, inputs_onnx)
probs = softmax(res[0], axis=1)
probs
'''