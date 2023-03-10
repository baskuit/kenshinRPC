import grpc
import torch
import torch.nn.functional as F
import time
import numpy as np
from concurrent import futures

import worker_pb2
import worker_pb2_grpc

# Define the maximum message length (in bytes) that the server can receive
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # 1 GB

class WorkerServicer(worker_pb2_grpc.WorkerServicer):
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.tensors = [None] * num_workers
        self.num_received = 0

    def SendTensor(self, request, context):
        worker_id = request.worker_id
        tensor_data = request.tensor_data

        # Convert the tensor data into a PyTorch tensor
        tensor = torch.from_numpy(np.frombuffer(tensor_data, dtype=np.float32))
        tensor = tensor.view(-1, 3, 32, 32)

        # Store the tensor and update the number of received tensors
        self.tensors[worker_id] = tensor
        self.num_received += 1

        # If all tensors have been received, take the average and send it back to the workers
        if self.num_received == self.num_workers:
            # Calculate the average tensor
            average_tensor = sum(self.tensors) / self.num_workers

            # Convert the PyTorch tensor into a bytes object
            average_tensor_data = average_tensor.numpy().tobytes()

            # Create the response object
            response = worker_pb2.AverageTensorResponse(average_tensor_data=average_tensor_data)

            # Reset the counters and the tensor list for the next iteration
            self.num_received = 0
            self.tensors = [None] * self.num_workers

            return response


def serve(num_workers):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)
    ])
    worker_pb2_grpc.add_WorkerServicer_to_server(WorkerServicer(num_workers), server)
    server.add_insecure_port('[::]:50051')
    server.start()

    print("Server started")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve(3)
