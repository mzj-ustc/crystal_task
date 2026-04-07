#!/usr/bin/env python

from CNAG1 import *
import yaml
import os
from optparse import OptionParser
from wae import *
from data_process import *
from utils import *
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import shutil as sh

parser = OptionParser()
parser.add_option("--configure", dest="configure", default="",
                  help="yaml.")
(options, args) = parser.parse_args()
configure=options.configure
with open(configure, encoding="utf-8") as f:
    temp = yaml.safe_load(f)

name = temp['name']
fin = temp['fin']
fout = temp['fout']
latent_space_zn = temp['latent']
neurons = temp['neurons']
BATCH_SIZE = temp['batch']
epoches = temp['epoches']
lr = temp['lr']
step_size = temp['step_size']
device_id = temp['device']
is_restart = temp['is_restart']
is_train = temp['is_train']
K_max = temp['Kmax']
alpha = temp['alpha']
input_f_dim=temp['input_f_dim']

print("Loading data ...")
data_loader,in_features,out_features=load_data(fin,fout,BATCH_SIZE)

in_features=in_features.reshape(-1, input_f_dim)

device = torch.device("cuda:"+str(device_id) if torch.cuda.is_available() else "cpu")

shape_input=out_features.shape[1]*out_features.shape[2]

# Define a WAE model
WAE_test=WAE(shape_input, middle_dim = neurons, latent_dim = latent_space_zn, device = device).to(device)
beta1, beta2 = 0.5, 0.9
optimizer= optim.Adam(WAE_test.parameters(), lr = lr, betas = (beta1, beta2))
scheduler = StepLR(optimizer, step_size=400, gamma=0.5)

date = datetime.now()
datestr = date.strftime("%y-%m-%d-%H-%M")
try:
    os.mkdir('saves')
except FileExistsError:
    pass
flog = './saves/' + name + '_' + datestr + '.log'
flog_id = open(flog, 'w')
sh.copy(configure, './saves/' + name + '_' + datestr + '.yaml')

if is_restart:
    # restart from a saved model
    fsave = find_latest(name)
    print(f"Load model from {fsave}")
    checkpoint = torch.load(fsave)
    WAE_test.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch_tot = checkpoint['epoch_tot']
else:
    # initialize a new run
    print("Initialize model")
    initialize_weights(WAE_test)
    epoch_tot = 0

#print(input_dim)
print("WAE archtecture:")
flog_id.writelines("WAE archtecture:\n")
flog_id.writelines(f"Latent dimension: {latent_space_zn}\n")
flog_id.writelines(f"neurons: {neurons}\n")
flog_id.writelines("YAML file:\n")
flog_id.writelines(repr(temp))
flog_id.writelines("\n")

print(WAE_test)
flog_id.writelines(repr(WAE_test))
#summary(WAE_test, (1, input_dim, input_dim))
total_params = sum(p.numel() for p in WAE_test.parameters())
print(f"Number of parameters: {total_params}")
flog_id.writelines(f"Number of parameters: {total_params}\n")
#exit()

print("Start training ...")

time_tot = 0
for epoch in range(epoches):
    episodic_loss_tot = []
    episodic_loss_reconstruction = []
    episodic_loss_MMD = []

    start_time = time.time()
    for idx, batch in enumerate(data_loader):
        # trigue train mode
        WAE_test.train()
        WAE_test.zero_grad()
        optimizer.zero_grad()

        # real samples: x, fake samples: x_fake, latent_space: z
        x = batch.to(device)
        x_fake, z = WAE_test(x)

        # total loss, reconstruction loss, MMD loss
        loss_tot, loss_reconstruction, loss_MMD = WAE_test.loss_func(x_fake, z, x)

        loss_tot.backward()
        #loss_reconstruction.backward()
        optimizer.step()

        episodic_loss_tot.append(loss_tot.item())
        episodic_loss_reconstruction.append(loss_reconstruction.item())
        episodic_loss_MMD.append(loss_MMD.item())

    time_tot += time.time() - start_time
    strs ="Episode: {}, total loss: {}, reconstruction loss: {}, MMD loss: {}, time: {} seconds".format(epoch + epoch_tot, np.array(episodic_loss_tot).mean(), np.array(episodic_loss_reconstruction).mean(), np.array(episodic_loss_MMD).mean(), time.time() - start_time) 
    print(strs)
    flog_id.writelines(strs + '\n')
    scheduler.step()

epoch_tot += epoches
torch.save({
    'epoch_tot': epoch_tot,
    'model_state_dict': WAE_test.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss_total': loss_tot,
    'loss_reconstruction': loss_reconstruction,
    'loss_MMD': loss_MMD,
    }, './saves/' + name + '_' + datestr + '.pt')

print(f"Finished {epoches} epoches for a total of {epoch_tot}.")
flog_id.writelines(f"Finished {epoches} epoches for a total of {epoch_tot}.\n")
print(f"Using {time_tot} seconds.")
flog_id.writelines(f"Using {time_tot} seconds.\n")

# This part will be in the evaluation eval.py [Todo]
# crystal=torch.tensor(out_features,dtype=torch.float)
# ae_final=WAE_test.encode(crystal.to('cpu')).cpu().detach().numpy()

# K=CNAK(np.array(ae_final),name,alpha=alpha,gamma=0.2,K_min=2,K_max=K_max)

# labels,labels1=process_result(ae_final,in_features,K)
    
# plt.scatter(in_features[:,1],in_features[:,0],c=labels,s = 5,cmap='Paired')
# plt.savefig('./LJ.jpg',dpi=200)
