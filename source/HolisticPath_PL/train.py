from static_params import *
from models import *

def train(model, train_dataloader, valid_dataloader, nr_epochs=c('num_local_epochs')):
    trainer = get_trainer()

    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader,
    )

    return trainer

def validate(model, valid_dataloader):
    trainer = get_trainer()
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

def test(model, test_dataloader):
    trainer = get_trainer()
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    pprint(test_metrics)
    return test_metrics

def get_trainer():
    return pl.Trainer(
        accelerator= 'gpu',
        max_epochs=c('num_local_epochs'),
        #devices=[c('gpu_id')],
    )

def net_instance(name):
    print(f"🌻 Created new model - {name} 🌻")
    return PT_Model()

def get_parameters(net, cid):
    print(f"⤺ Get model parameters of client {cid}")
    return [val.cpu().numpy() for _, val in net.model.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray], cid):
    print(f"⤻ Set model parameters of client {cid}")
    params_dict = zip(net.model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )
    net.model.load_state_dict(state_dict, strict=True)

def save_model(net, name):
    print(f"🔒 Saved the model of client {name} to the disk. 🔒")
    torch.save(net.model.state_dict(), f"{name}.pth")

def load_model(name):
    print(f"🛅 Loaded the model of client {name} from the disk. 🛅")
    net = net_instance(f"{name}")
    net.model.load_state_dict(torch.load(f"{name}.pth"))
    return net