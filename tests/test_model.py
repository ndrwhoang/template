"""
with torch.autograd.set_detect_anomaly(True):
    for i, batch in enumerate(dataloader):
        model.zero_grad(set_to_none=True)
        model_output = model(**batch)
        model_output.loss.backward()
        optimizer.step()

"""
