import torch
from transformer import TextGenerator, create_data, load_checkpoint, eval_loss
from tokenizer import Tokenizer


model_path = "models/checkpoint_3000.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextGenerator().to(device)
model.load_state_dict(torch.load((model_path), weights_only=True)["model"])
model.eval()

Xtr, Ytr, Xval, Yval = create_data()

Xtr = Xtr.to(device)
Xval = Xval.to(device)

print("----- Model used    -----")
print(f"{model_path=}")
print("----- Generate text -----")
print("")
print("--- Training ---")
print(model.generate(torch.zeros(Xtr[:1].shape).int()))
print("")
print("--- Validation ---")
print(model.generate(Xval[:1]))
print("")
