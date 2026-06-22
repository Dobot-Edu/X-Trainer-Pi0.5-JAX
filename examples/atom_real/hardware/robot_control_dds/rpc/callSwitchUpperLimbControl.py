import socket
import json
import time

def recv_all(sock, buffer_size=1024):
    data = b''
    while True:
        part = sock.recv(buffer_size)
        if not part:
            break
        data += part
        if len(part) < buffer_size:
            break
    return data


class JsonRpcClient:
    def __init__(self, ip='192.168.8.234', port=51234):
        self.server_ip = ip
        self.server_port = port

    def CallSwitchUpperLimbControl(self, is_on: bool):
        request = {
            "jsonrpc": "2.0",
            "method": "SwitchUpperLimbControl",
            "params": {"is_on": is_on},
            "id": 1
        }

        request_str = json.dumps(request)

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.server_ip, self.server_port))
                s.sendall(request_str.encode())

                response_bytes = recv_all(s)
                response_str = response_bytes.decode()
                response = json.loads(response_str)

                if "result" in response:
                    print(f"SwitchUpperLimbControl Result: {response['result']}")
                    return response['result']
                elif "error" in response:
                    print(f"RPC Error: {response['error']['message']}")
                    return -1
                else:
                    print("Unexpected response format")
                    return -1

        except Exception as e:
            print(f"Exception: {e}")
            return -1


if __name__ == "__main__":
    rpc = JsonRpcClient()
    while 1:
        # rpc.CallSwitchUpperLimbControl(True)
        # time.sleep(2)
        rpc.CallSwitchUpperLimbControl(False)
        time.sleep(2)