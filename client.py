import grpc
import torch
import numpy as np

import worker_pb2
import worker_pb2_grpc

def run():
    # Connect to the server
    channel = grpc.insecure_channel('localhost:50051')
    stub = worker_pb2_grpc.WorkerStub(channel)

    # Create the PyTorch tensor
    tensor = torch.randn(3, 32, 32)
    tensor_data = tensor.numpy().tobytes()

    # Create the request object and send the tensor to the server
    request = worker_pb2.TensorRequest(worker_id=0, tensor_data=tensor_data)
    response = stub.SendTensor(request)

    # Convert the response data into a PyTorch tensor
    average_tensor_data = response.average_tensor_data
    average_tensor = torch.from_numpy(np.frombuffer(average_tensor_data, dtype=np.float32))
    average_tensor = average_tensor.view(-1, 3, 32, 32)

    # Print the average tensor
    print('Average tensor:', average_tensor)

if __name__ == '__main__':
    run()
