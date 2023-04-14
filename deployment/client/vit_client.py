import tritonclient.grpc as triton_grpc
import tritonclient.utils
import numpy as np

def main():
    grpc_client = triton_grpc.InferenceServerClient(
        url='localhost:8001',
        verbose = False
        )

    triton_input_grpc = [
        triton_grpc.InferInput('tokens', (4, 3, 384, 384), tritonclient.utils.np_to_triton_dtype(np.float32)),
    ]
    triton_input_grpc[0].set_data_from_numpy(np.random.rand(4, 3, 384, 384).astype(np.float32))
    triton_output_grpc = triton_grpc.InferRequestedOutput('logits')
    request_grpc = grpc_client.infer(
            'vit_trt',
            model_version='1',
            inputs=triton_input_grpc,
            outputs=[triton_output_grpc]
        )
    outt = request_grpc.as_numpy('logits')

if __name__ == "__main__":
    main()