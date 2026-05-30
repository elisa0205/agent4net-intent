import os
import uuid
import subprocess
import tempfile
from typing import Optional

class KindCluster:
    def __init__(self, name: Optional[str] = None, config: Optional[str] = None, wait: str = "90s"):
        self.name = name or f"kind-cluster-{uuid.uuid4().hex[:6]}"
        self.config = config
        self.wait = wait
        self.kubeconfig = os.path.join(tempfile.gettempdir(), f"{self.name}-kubeconfig.yaml")

    def create(self, timeout: int = 240):
        # kind create cluster --name provider-germany --config $PWD/provider-cluster-config.yaml --kubeconfig $PWD/provider-DE-config.yaml
        cmd = ["kind", "create", "cluster", "--name", self.name, "--kubeconfig", self.kubeconfig, "--wait", self.wait]
        if self.config:
            cmd.extend(["--config", self.config])
        res = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)
        return res
    
    def apply(self, manifest_path: str, timeout: int = 120):
        cmd = ["kubectl", "apply", "-f", manifest_path, "--kubeconfig", self.kubeconfig]
        res = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)
        return res
    
    def delete(self, timeout: int = 120):
        cmd = ["kind", "delete", "cluster", "--name", self.name]
        res = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)
        try:
            if os.path.exists(self.kubeconfig):
                os.remove(self.kubeconfig)
        except OSError:
            pass
    
    def __enter__(self):
        try:
            self.create()
            return self
        except Exception:
            self.delete()
            raise

    def __exit__(self, exc_type, exc, tb):
        self.delete()