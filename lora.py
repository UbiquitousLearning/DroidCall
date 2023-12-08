import torch
from torch import nn

from typing import List

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

class LoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.A = nn.Linear(self.linear.in_features, r, bias=False)
        self.A.weight.data.normal_(0, 1)
        self.B = nn.Linear(r, self.linear.out_features, bias=False)
        self.B.weight.data.zero_()
    
    def freeze_linear(self):
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        out = self.linear(x)
        delta = self.B(self.A(x))
        return out + (self.alpha/self.r) * delta
    
def replace_linear(model: nn.Module, r: int, alpha: float, target_modules: List[str] = ["fc"], freeze: bool = True, verbose: bool = False):
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, nn.Linear):
                parent, target, target_name = _get_submodules(model, name)
                if verbose:
                    print(f"Replacing {name} with LoRA  freezed: {freeze}")
                l = LoRA(module, r, alpha)
                setattr(parent, target_name, l)
                if freeze:
                    l.freeze_linear()
    return model
                    
def freeze_all_LoRA(model: nn.Module, verbose: bool = False):
    for name, module in model.named_modules():
        if isinstance(module, LoRA):
            if verbose:
                print(f"Freezing {name}")
            module.freeze_linear()
    

def print_cuda_mem(str: str = ""):
    print(f"-------------------{str}-------------------")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    print(f"memory_reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")
    print(f"max_memory_allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.1f} MB")
    print(f"max_memory_reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2:.1f} MB")
    print("--------------------------------------------")
    
class MyModule(nn.Module):
    def __init__(self, num_hidden_layers: int, in_features: int, out_features: int, hidden_features: int):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.in_features = in_features
        self.out_features = out_features
        self.fc_in = nn.Linear(in_features, hidden_features)
        self.fc_out = nn.Linear(hidden_features, out_features)
        self.fcs = nn.ModuleList([nn.Linear(hidden_features, hidden_features, bias=False) for _ in range(num_hidden_layers)])
        
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, 0, 1)
            elif "bias" in name:
                nn.init.zeros_(param)
                
    def forward(self, x):
        out = self.fc_in(x)
        for fc in self.fcs:
            out = fc(out)
        out = self.fc_out(out)
        return out
    
def test_model(model: nn.Module):
    print(f"current device: {torch.cuda.current_device()}")
    print(model)
    print_cuda_mem("before move to cuda")
    
    model.cuda()
    print_cuda_mem("after move to cuda")
    
    batch_size = 1024
    x = torch.randn(batch_size, 28*28).cuda()
    print_cuda_mem(f"create x of shape{x.shape}")
    
    out = model(x)
    print_cuda_mem(f"after forward pass")
    
    out.sum().backward()
    print_cuda_mem(f"after backward pass")
    
    del out, x
    torch.cuda.empty_cache()

if __name__ == "__main__":
    model = MyModule(10, 28*28, 10, 4096)
    # replace_linear(model, 8, 0.1,freeze=True ,verbose=True)
    test_model(model)
    del model
    torch.cuda.empty_cache()
    print_cuda_mem("after delete model and other data")
