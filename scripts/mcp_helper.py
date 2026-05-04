#!/usr/bin/env python3
"""
Direct MCP client for the ComfyUI agent.
Spawns the MCP server and calls tools via JSON-RPC over stdio.
"""
import json
import subprocess
import sys
import uuid

SERVER_CMD = ["/home/ubuntu/Documents/programs/Comfy-Cozy/.venv/bin/python", "-m", "agent.mcp_server"]

class ComfyMCPClient:
    def __init__(self):
        self.proc = subprocess.Popen(
            SERVER_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._readline()  # skip first status line if any
        self._send_init()

    def _send(self, msg: dict) -> None:
        raw = json.dumps(msg) + "\n"
        self.proc.stdin.write(raw)
        self.proc.stdin.flush()

    def _readline(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            return {}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            # skip non-JSON log lines
            return self._readline()

    def _send_init(self):
        self._send({
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pose-director-helper", "version": "1.0.0"},
            },
        })
        # read init response
        while True:
            resp = self._readline()
            if resp.get("id"):
                break
        # send initialized notification
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, tool_name: str, arguments: dict) -> dict:
        req_id = str(uuid.uuid4())
        self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        while True:
            resp = self._readline()
            if resp.get("id") == req_id:
                return resp

    def list_tools(self) -> list:
        req_id = str(uuid.uuid4())
        self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/list",
            "params": {},
        })
        while True:
            resp = self._readline()
            if resp.get("id") == req_id:
                result = resp.get("result", {})
                return result.get("tools", [])

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)


def main():
    client = ComfyMCPClient()
    tools = client.list_tools()
    print(f"Connected. {len(tools)} tools available.")
    for t in tools:
        print(" -", t["name"])
    client.close()


if __name__ == "__main__":
    main()
