import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/VMS',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=249,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,help='')
parser.add_argument('--plotheatmap',type=str,default='True',help='')

args = parser.parse_args()

def main():
    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    model =  gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    print('model load successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device) # (4281, 12, 249, 2) (1, 12, 249, 2)
    realy = realy.transpose(1,3)[:,0,:,:] # (4281, 1, 249, 12)

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()): # batch
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze())
        print("batch:", outputs[-1].shape) # (batch, 249, 12) ????

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...] # (4281, 249, 12) : sample, node, time

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i]) # (sample, node, time_step)
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))


    if args.plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp*(1/np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig("./emb"+ '.pdf')

    yhat = scaler.inverse_transform(yhat[:,:,:]).cpu().detach().numpy()
    print("sample, num_nodes, time_step: ", yhat.shape) # (sample, num_nodes, time_step)

    sample = yhat.shape[0]
    yhat_2d = yhat[0,:,:]
    for i in range(1, sample):
        yhat_2d = np.hstack((yhat_2d, yhat[i, :, :]))
    # time_step, num_nodes:  (2, 249, 12)
    result = yhat_2d.transpose(1, 0)
    print(yhat.shape, yhat_2d.shape, result.shape)
    df = pd.read_hdf("data/vms_test_data.h5")
    df.iloc[:, :] = result
    df.index = df.index + pd.to_timedelta('1H')
    df.to_hdf("./vms_timetable.h5", key="df")
    # df2.to_csv('./wave.csv',index=False)

if __name__ == "__main__":
    main()