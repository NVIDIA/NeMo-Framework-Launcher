import tritonclient.grpc as triton_grpc
import tritonclient.utils
import numpy as np

def main():
    grpc_client = triton_grpc.InferenceServerClient(
        url='localhost:8001',
        verbose = False
        )
    prompt = 'A photo of a Shiba Inu dog with a backpack riding a bike. It is wearing sunglasses and a beach hat.'
    triton_input_grpc = [
        triton_grpc.InferInput('prompt', (1,), tritonclient.utils.np_to_triton_dtype(np.object_))
    ]
    triton_input_grpc[0].set_data_from_numpy(np.array(prompt, dtype=np.object_).reshape(1))
    triton_output_grpc = triton_grpc.InferRequestedOutput('generated_image')
    request_grpc = grpc_client.infer(
            'stable_diffusion',
            model_version='1',
            inputs=triton_input_grpc,
            outputs=[triton_output_grpc]
        )
    outt = request_grpc.as_numpy('generated_image')

if __name__ == "__main__":
    main()