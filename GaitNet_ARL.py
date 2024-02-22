import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gait_conv import GaitConv
from torch_scatter import scatter_mean
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt
import os
import pandas as pd
from gait_env import GaitEnv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gym
from datetime import datetime
import pickle


# Define CAI Dataset.
class CAIDataset(Dataset):
    def __init__(self, num_phases):
        with open(rf'Data\{dataName}.pkl', 'rb') as file:
            NEU_CAI = pickle.load(file)

        # Node features.
        self.data = NEU_CAI['data']
        # Cycle labels, 1: CAI, 0: Health.
        self.labels = NEU_CAI['labels']
        # Cycle genders, 1: female, 0: male.
        self.genders = NEU_CAI['genders']
        # Subject genders, 1: female, 0: male.
        self.subject_genders = NEU_CAI['subject_genders']
        # Cycle index.
        self.cycleIdx = NEU_CAI['cycleIdx']
        # Number of subjects.
        self.num_subjects = NEU_CAI['num_subjects']
        # Number of cycles.
        self.num_cycles = NEU_CAI['num_cycles']
        # Node types, i.e. nodes of x,y,z types.
        self.node_types = NEU_CAI['node_types']
        # Edge indexs by creating kNN-Graph (Graph connectivity in COO format with shape [2, num_edges]).
        self.edge_indexs = NEU_CAI['edge_indexs']
        # Edge features.
        self.edge_features = NEU_CAI['edge_features']
        # Edge types, i.e. 1: intra-joint, 0: inter-joint.
        self.edge_types = NEU_CAI['edge_types']

        # Set number of phases.
        self.num_phases = num_phases
        # Set phase's frame-index.
        phase_frame_idx = np.array([i / num_phases * 84 for i in range(num_phases + 1)]).round().astype(int)
        self.phase_frame_idx = phase_frame_idx.repeat(2)[1:-1].reshape(-1, 2)

        # Set initial attention on frame index.
        self.reset()


    def reset(self):
        # Pay equal attention to each frame.
        self.phase_atts = torch.ones(self.num_cycles, self.num_phases) / self.num_phases
        self.frame_atts = torch.ones(self.num_cycles, 84) / 84

        # # Pay distinct attention to each frame.
        # self.phase_atts = torch.ones(self.num_cycles, self.num_phases)
        # self.frame_atts = torch.ones(self.num_cycles, 84)
        # phase_att = torch.ones(self.num_phases)
        # phase_att[[2, 7]] = 3
        # phase_att[[1, 3, 6, 8]] = 2
        # phase_att /= phase_att.sum()
        # self.phase_atts[:] = phase_att
        # for k, (i, j) in enumerate(self.phase_frame_idx):
        #     self.frame_atts[:, i:j] = phase_att[k] / (j - i)


    def __len__(self):
        return len(self.labels)


    def select_key_phases(self, input, pre_att):
        # Apply policy to select key phases
        obs = input.unsqueeze(0).unsqueeze(0)
        att, _state = agent.predict(obs, deterministic=True)
        att = torch.tensor(att).squeeze()
        if att.sum() == 0:
            att[0] = 1
            att[5] = 1
        att = att / att.sum()
        att = att_mmt * pre_att + att
        att = att / att.sum()

        return att


    def __getitem__(self, idx):

        node_feature = torch.from_numpy(self.data[idx]).type(torch.float).T
        node_type = torch.tensor(self.node_types[idx], dtype=torch.long)
        edge_index = torch.tensor(self.edge_indexs[idx]).type(torch.long)
        edge_feature = torch.tensor(self.edge_features[idx]).type(torch.float)
        edge_type = torch.tensor(self.edge_types[idx]).type(torch.long)
        label = torch.tensor(self.labels[idx]).type(torch.long)
        gender = torch.tensor(self.genders[idx]).type(torch.long)

        if promo == 0 or epoch > 0 or is_DRL:
            phase_att = self.phase_atts[idx]
            frame_att = self.frame_atts[idx]
        else:
            phase_att = self.select_key_phases(node_feature, self.phase_atts[idx])
            self.phase_atts[idx] = phase_att
            for k, (i, j) in enumerate(self.phase_frame_idx):
                self.frame_atts[idx, i:j] = phase_att[k] / (j - i)
            frame_att = self.frame_atts[idx]

        data = Data(x=node_feature, phase_att=phase_att, frame_att=frame_att, key_frames=node_feature*frame_att, node_type=node_type, edge_index=edge_index, edge_feature=edge_feature, edge_type=edge_type, y=label, gender=gender, cycle=idx)

        return data


# Define GaitNet.
class GaitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GaitConv(in_channels=84, out_channels=32, kernel_size=9, num_node_types=3, num_edge_types=2, edge_feature_dim=1, edge_feature_emb_dim=21, edge_type_emb_dim=5)
        self.conv2 = GaitConv(in_channels=32, out_channels=2, kernel_size=9, num_node_types=3, num_edge_types=2, edge_feature_dim=1, edge_feature_emb_dim=8, edge_type_emb_dim=2)

    def forward(self, data):
        x, node_type = data.key_frames, data.node_type
        edge_index, edge_feature, edge_type = data.edge_index, data.edge_feature, data.edge_type
        batch = data.batch

        x = self.conv1(x=x, edge_index=edge_index, node_type=node_type, edge_feature=edge_feature, edge_type=edge_type)
        x = F.elu(x)
        x = self.conv2(x=x, edge_index=edge_index, node_type=node_type, edge_feature=edge_feature, edge_type=edge_type)

        if batch is None:
            x = x.mean(dim=0)
        else:
            x = scatter_mean(x, batch, dim=0)

        return x



# Define policy network.
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 96):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(1, 3), stride=(1, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(3, 3), stride=(3, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)

        return x



# GaitNet training process
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    accuracy = 0
    female_accuracy = 0
    female_size = 0
    male_accuracy = 0
    male_size = 0
    for batch, data in enumerate(dataloader):
        data = data.to(device)
        y = data.y
        female_idx = data.gender.type(torch.bool)
        male_idx = ~ female_idx

        # Compute prediction error
        pred = model(data)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        corrects = (pred.argmax(dim=1) == y).type(torch.float)
        accuracy += corrects.sum().item()
        female_accuracy += corrects[female_idx].sum().item()
        female_size += female_idx.type(torch.long).sum().item()
        male_accuracy += corrects[male_idx].sum().item()
        male_size += male_idx.type(torch.long).sum().item()

    accuracy /= size
    female_accuracy /= female_size
    male_accuracy /= male_size

    return accuracy, female_accuracy, male_accuracy



# GaitNet testing process
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, accuracy = 0, 0
    female_accuracy = 0
    female_size = 0
    male_accuracy = 0
    male_size = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            y = data.y
            female_idx = data.gender.type(torch.bool)
            male_idx = ~ female_idx

            # Compute prediction
            pred = model(data)
            test_loss += loss_fn(pred, y).item()

            corrects = (pred.argmax(dim=1) == y).type(torch.float)
            accuracy += corrects.sum().item()
            female_accuracy += corrects[female_idx].sum().item()
            female_size += female_idx.type(torch.long).sum().item()
            male_accuracy += corrects[male_idx].sum().item()
            male_size += male_idx.type(torch.long).sum().item()

    test_loss /= num_batches

    accuracy /= size

    if female_size > 0:
        female_accuracy /= female_size
    else:
        female_accuracy = np.nan

    if male_size > 0:
        male_accuracy /= male_size
    else:
        male_accuracy = np.nan

    return accuracy, female_accuracy, male_accuracy



# Plot GaitNet+DRL results.
def plot_results():
    # Plot learning curve.
    trainAccr_mean = np.mean(train_accuracy, axis=0)
    trainAccr_std = np.std(train_accuracy, axis=0)
    testAccr_mean = np.mean(test_accuracy, axis=0)
    testAccr_std = np.std(test_accuracy, axis=0)

    path = rf'Result\Learning Curve'
    os.makedirs(path, exist_ok=True)

    # Plot learning curve for each promotion separately.
    epochs = np.arange(num_epochs)
    for p in range(num_promos):
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.rcParams.update({'font.size': 20})
        # plt.errorbar(epochs, trainAccr_mean[p], yerr=trainAccr_std[p], label=f'Train (Max={max(trainAccr_mean[p]):0.2f})')
        plt.plot(epochs, trainAccr_mean[p], label=f'Train (Max={max(trainAccr_mean[p]):0.2f})')
        # plt.errorbar(epochs, testAccr_mean[p], yerr=testAccr_std[p], label=f'Test (Max={max(testAccr_mean[p]):0.2f})')
        plt.plot(epochs, testAccr_mean[p], label=f'Test (Max={max(testAccr_mean[p]):0.2f})')
        plt.legend()
        plt.title(f'{modelName}, Promo{p}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(epochs, epochs)
        filename = f'LearningCurve_Promo{p}.jpg'
        plt.savefig(os.path.join(path, filename))
        plt.close()

    # Plot learning curve for all promotions.
    epochs = np.arange(testAccr_mean.size)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.rcParams.update({'font.size': 20})
    # plt.errorbar(epochs, trainAccr_mean.reshape(-1), yerr=trainAccr_std.reshape(-1), label=f'Train (Max={max(trainAccr_mean.reshape(-1)):0.2f})')
    plt.plot(epochs, trainAccr_mean.reshape(-1), label=f'Train (Max={max(trainAccr_mean.reshape(-1)):0.2f})')
    # plt.errorbar(epochs, testAccr_mean.reshape(-1), yerr=testAccr_std.reshape(-1), label=f'Test (Max={max(testAccr_mean.reshape(-1)):0.2f})')
    plt.plot(epochs, testAccr_mean.reshape(-1), label=f'Test (Max={max(testAccr_mean.reshape(-1)):0.2f})')
    plt.legend()
    plt.title(f'{modelName}, All Promotions')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs, epochs)
    filename = f'LearningCurve_All.jpg'
    plt.savefig(os.path.join(path, filename))
    plt.close()

    # Save train accuracy into Excel file.
    filename = os.path.join(path, 'Train_Accuracy.xlsx')
    with pd.ExcelWriter(filename) as writer:
        for p in range(num_promos):
            df = pd.DataFrame(train_accuracy[:,p,:])
            df.to_excel(writer, sheet_name=f'Promo_{p}')

    # Save test accuracy into Excel file.
    filename = os.path.join(path, 'Test_Accuracy.xlsx')
    with pd.ExcelWriter(filename) as writer:
        for p in range(num_promos):
            df = pd.DataFrame(test_accuracy[:,p,:])
            df.to_excel(writer, sheet_name=f'Promo_{p}')


    # Plot accuracy bar for female and male subjects.
    train_female_accuracy_mean = np.mean(train_female_accuracy, axis=0)
    train_male_accuracy_mean = np.mean(train_male_accuracy, axis=0)
    test_female_accuracy_mean = np.nanmean(test_female_accuracy, axis=0)
    test_male_accuracy_mean = np.nanmean(test_male_accuracy, axis=0)

    path = rf'Result\Accuracy Bar'
    os.makedirs(path, exist_ok=True)

    # Set width of bar.
    barWidth = 0.2
    # Set position of bar on X-axis.
    br1 = np.arange(5)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the bar plot.
    for p in range(num_promos):
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.rcParams.update({'font.size': 20})
        plt.bar(br1, train_female_accuracy_mean[p][-5:], color='r', width=barWidth, edgecolor='grey', label='Train Female')
        plt.bar(br2, train_male_accuracy_mean[p][-5:], color='g', width=barWidth, edgecolor='grey', label='Train Male')
        plt.bar(br3, test_female_accuracy_mean[p][-5:], color='b', width=barWidth, edgecolor='grey', label='Test Female')
        plt.bar(br4, test_male_accuracy_mean[p][-5:], color='c', width=barWidth, edgecolor='grey', label='Test Male')
        plt.legend()
        plt.title(f'{modelName}, Promo{p}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.xticks([t + 1.5 * barWidth for t in range(5)], np.arange(num_epochs)[-5:])
        filename = f'AccuracyBar_Promo{p}.jpg'
        plt.savefig(os.path.join(path, filename))
        plt.close()



# Plot DRL computed frame attentions.
def plot_frame_attentions(frame_idx, frame_att_mean, frame_att_std, path, plot_errorbar):

    tick = np.array([i/10*84 for i in range(11)]).round().astype(int)
    tick_label = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']

    # Plot DRL computed frame attention bar for each promotion separately.
    for p in range(num_promos):
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.rcParams.update({'font.size': 20})
        plt.bar(frame_idx, frame_att_mean[p])
        plt.title(f'Frame Attention Bar, Promo{p}')
        plt.xticks(tick, tick_label)
        plt.xlabel('Frame Index')
        plt.ylabel('Attention')
        filename = f'FrameAtt_Bar_Promo{p}.jpg'
        plt.savefig(os.path.join(path, filename))
        plt.close()

    if plot_errorbar:
        # Plot DRL computed frame attention bar and error-bar for each promotion separately.
        for p in range(num_promos):
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.rcParams.update({'font.size': 20})
            plt.bar(frame_idx, frame_att_mean[p])
            plt.errorbar(frame_idx, frame_att_mean[p], yerr=frame_att_std[p], ecolor='r')
            plt.title(f'Frame Attention, Promo{p}')
            plt.xticks(tick, tick_label)
            plt.xlabel('Frame Index')
            plt.ylabel('Attention')
            filename = f'FrameAtt_Promo{p}.jpg'
            plt.savefig(os.path.join(path, filename))
            plt.close()

        # Plot DRL computed frame attention error-bar for each promotion separately.
        for p in range(num_promos):
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.rcParams.update({'font.size': 20})
            plt.errorbar(frame_idx, frame_att_mean[p], yerr=frame_att_std[p], ecolor='r')
            plt.title(f'Frame Attention Error Bar, Promo{p}')
            plt.xticks(tick, tick_label)
            plt.xlabel('Frame Index')
            plt.ylabel('Attention')
            filename = f'FrameAtt_ErrorBar_Promo{p}.jpg'
            plt.savefig(os.path.join(path, filename))
            plt.close()



# Plot DRL selected key frames.
def plot_key_frames():

    frame_idx = np.arange(84)

    # Plot all subjects frame attentions.
    frame_att_mean = np.mean(frame_attentions, axis=(0, 2))
    frame_att_std = np.std(frame_attentions, axis=(0, 2))
    path = rf'Result\Key Frames'
    os.makedirs(path, exist_ok=True)
    plot_frame_attentions(frame_idx, frame_att_mean, frame_att_std, path, True)


    # Save frame attentions.
    filename = os.path.join(path, 'Frame_Attentions.xlsx')
    frame_att_mean = np.mean(frame_attentions, axis=2)
    with pd.ExcelWriter(filename) as writer:
        for p in range(num_promos):
            df = pd.DataFrame(frame_att_mean[:, p, :])
            df.to_excel(writer, sheet_name=f'Promo_{p}')

    filename = os.path.join(path, 'Frame_Attentions.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(frame_attentions, file)


    # # Plot individual subject frame attentions.
    # for i in range(num_folds):
    #     frame_att = frame_attentions[:, :, cycleIdx[i][0]:cycleIdx[i][1], :]
    #     frame_att_mean = np.mean(frame_att, axis=(0, 2))
    #     frame_att_std = np.std(frame_att, axis=(0, 2))
    #     path = rf'Result\Key Frames\{i}'
    #     os.makedirs(path, exist_ok=True)
    #     plot_frame_attentions(frame_idx, frame_att_mean, frame_att_std, path, False)





#============================== MAIN ==============================
if __name__ == '__main__':

    # Start timer.
    START_TIME = datetime.now()


    # Define data name.
    dataName = 'NEU_CAI'
    print(dataName)

    # Define model name. DRL version: Soft Attention, Dependent Promotions.
    modelName = 'Key_Phases_MultiBinary'

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Load data from CAI dataset.
    dataset = CAIDataset(num_phases=10)

    # Set number of folds for leave-one-subject-out cross validation.
    num_folds = dataset.num_subjects

    # Get number of cycles.
    num_cycles = dataset.num_cycles

    # Get start and end cycle index of subjects.
    cycleIdx = dataset.cycleIdx

    # Get number of phases.
    num_phases = dataset.num_phases

    # Get phase's frame-index.
    phase_frame_idx = dataset.phase_frame_idx

    # Set number of DRL promotions.
    num_promos = 1

    # Set number of GaitNet epochs.
    num_epochs = 5

    # Set mini-batch size.
    #batch_sizes = [16, 32, 64, 96, 100, 128, 200, 256]
    batch_size = 16

    # Set regularizer.
    #weight_decays = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    weight_decays = [0]

    # Set number of DRL batches to be trained.
    num_batches = 10

    # Set number of DRL environments to be trained.
    num_envs = 10

    # Set DRL reward.
    # reward_set=[0.1, 1, 10, 100]
    reward_set = [1]

    # Set value function coefficient for the DRL loss calculation.
    # vf_coefs = [1, 0.5, 0.1, 0.01, 0.001, 0.0001]
    vf_coef = 0.5

    # Set attention momentum coefficient for the DRL selected key frames.
    # att_mmt_set = [0.0, 0.5, 0.9]
    att_mmt_set = [0.0]

    # Set number of DRL epochs.
    num_outer_DRL_epochs = 1
    num_inner_DRL_epochs = 5


    for reward in reward_set:
        for att_mmt in att_mmt_set:
            for weight_decay in weight_decays:
                print('='*20)
                print(f'Reward {reward}, Attention Momentum {att_mmt}, Weight Decay {weight_decay}')
                print('='*20)

                # Set version name.
                versionName = f'{num_promos}promos_{num_epochs}epochs_{batch_size}batchSize_{num_batches}drlBatches_{reward}reward_{att_mmt}attMmt_{num_outer_DRL_epochs}outDRLepochs_{num_inner_DRL_epochs}inDRLepochs_{num_envs}envs'

                # Initialize total train and test accuracy for folds.
                train_accuracy = np.empty((num_folds, num_promos, num_epochs))
                test_accuracy = np.empty((num_folds, num_promos, num_epochs))

                # Initialize female train and test accuracy for folds.
                train_female_accuracy = np.empty((num_folds, num_promos, num_epochs))
                test_female_accuracy = np.empty((num_folds, num_promos, num_epochs))

                # Initialize male train and test accuracy for folds.
                train_male_accuracy = np.empty((num_folds, num_promos, num_epochs))
                test_male_accuracy = np.empty((num_folds, num_promos, num_epochs))

                # Initialize test majority voting accuracy for folds.
                test_MVA = np.empty((num_folds, num_promos, num_epochs))

                # Initialize frame attentions.
                frame_attentions = np.empty((num_folds, num_promos, num_cycles, 84))


                # Perform leave-one-subject-out cross validation.
                for fold in range(num_folds):
                    print('-----------------')
                    print(f'FOLD {fold}')
                    print('-----------------')

                    # Reset CAI dataset.
                    dataset.reset()

                    train_indices = list(range(0, cycleIdx[fold][0])) + list(range(cycleIdx[fold][1], num_cycles))
                    test_indices = list(range(cycleIdx[fold][0], cycleIdx[fold][1]))

                    # Subset of a dataset at specified indices.
                    train_dataset = Subset(dataset, train_indices)
                    test_dataset = Subset(dataset, test_indices)

                    # Create data loaders for GaitNet.
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

                    # Create data loader for DRL.
                    drl_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

                    # Create the GNN
                    model = GaitNet().to(device)

                    # Choose loss function and optimizer.
                    loss_fn = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

                    # Iteratively perform GaitNet and DRL to promote each other mutually.
                    for promo in range(num_promos):
                        # Perform GaitNet.
                        is_DRL = False
                        for epoch in range(num_epochs):
                            # Training
                            accuracy, female_accuracy, male_accuracy = train(train_dataloader, model, loss_fn, optimizer)
                            train_accuracy[fold, promo, epoch] = accuracy
                            train_female_accuracy[fold, promo, epoch] = female_accuracy
                            train_male_accuracy[fold, promo, epoch] = male_accuracy

                            # Testing
                            accuracy, female_accuracy, male_accuracy = test(test_dataloader, model, loss_fn)
                            test_accuracy[fold, promo, epoch] = accuracy
                            test_female_accuracy[fold, promo, epoch] = female_accuracy
                            test_male_accuracy[fold, promo, epoch] = male_accuracy
                            test_MVA[fold, promo, epoch] = round(accuracy)  # majority-voting accuracy


                        # Save frame attentions.
                        frame_attentions[fold, promo] = dataset.frame_atts


                        # Perform DRL.
                        if promo < num_promos-1:
                            is_DRL = True
                            for e in range(num_outer_DRL_epochs):
                                for batch, data in enumerate(drl_dataloader):
                                    if batch == num_batches:
                                        break
                                    data = data.to(device)
                                    envs = DummyVecEnv([lambda: GaitEnv(n_channels=18, n_frames=84, epi_len=1, data=data, model=model, att_mmt=att_mmt, reward=reward, num_phases=num_phases, phase_frame_idx=phase_frame_idx) for _ in range(num_envs)])
                                    if promo==0 and e==0 and batch==0:
                                        policy_kwargs = dict(
                                            features_extractor_class=CustomCNN,
                                            net_arch = [32],
                                            # net_arch = [128, dict(vf=[64, 32], pi=[256, 128])],
                                            activation_fn=torch.nn.ELU,
                                        )
                                        agent = PPO('CnnPolicy', envs, policy_kwargs=policy_kwargs, n_steps=1, batch_size=num_envs, n_epochs=num_inner_DRL_epochs, vf_coef=vf_coef, verbose=0)
                                        agent.learn(total_timesteps=num_envs)
                                        agent.save('ppo_gait')
                                    else:
                                        agent = PPO.load('ppo_gait', envs)
                                        agent.learn(total_timesteps=num_envs)
                                        agent.save('ppo_gait')

                # Plot GaitNet results.
                plot_results()

                # Plot DRL selected key frames results.
                plot_key_frames()


    # Stop timer.
    END_TIME = datetime.now()

    print(f"Elapsed Time: {END_TIME-START_TIME}")

