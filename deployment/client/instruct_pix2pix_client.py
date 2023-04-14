from PIL import Image
import tritonclient.grpc as triton_grpc
import tritonclient.utils
import numpy as np

def main():
    input_image = Image.open("/data/selene/example.jpg").convert("RGB")
    grpc_client = triton_grpc.InferenceServerClient(
        url='localhost:8001',
        verbose = False
        )
    prompt = "turn him into a cyborg"
    triton_input_grpc = [
        triton_grpc.InferInput('prompt', (1,), tritonclient.utils.np_to_triton_dtype(np.object_)),
        triton_grpc.InferInput('input_image', (512, 512, 3), tritonclient.utils.np_to_triton_dtype(np.uint8))
    ]
    triton_input_grpc[0].set_data_from_numpy(np.array(prompt, dtype=np.object_).reshape(1))
    triton_input_grpc[1].set_data_from_numpy(np.array(Image.open("/data/example.jpg").convert("RGB")))
    triton_output_grpc = triton_grpc.InferRequestedOutput('generated_image')
    request_grpc = grpc_client.infer(
            'instruct_pix2pix',
            model_version='1',
            inputs=triton_input_grpc,
            outputs=[triton_output_grpc]
        )
    outt = request_grpc.as_numpy('generated_image')

if __name__ == "__main__":
    main()