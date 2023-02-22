import torch
from tqdm import tqdm
def train(model,device):
    optimizer = torch.optim.AdamW(model.parameters(),lr = 0.001)
    # Construct data_loader, optimizer, etc.
    for i in tqdm(range(1000)):
        if(i == 999):
            print('YES')
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(torch.randn(12,200).to(device = device)), 
                                     torch.randn(12,1).to(device = device)).backward()
        optimizer.step()  # This will update the shared parameters
