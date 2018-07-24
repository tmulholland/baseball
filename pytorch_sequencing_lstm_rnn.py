import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np

class Sequence(nn.Module):
    """ Recurrent Neural Network (RNN) using Long-Short Term Memory (LSTM) hidden networks
    to encode a state of the sequence of pitches.
    """
    
    
    def __init__(self, hidden_dim, input_dim, num_pitches):
        """ Constructor 
        Args:
             Number of nodes in hidden layer,
             Number of inputs,
             Number of pitch types
        """
        
        super(Sequence, self).__init__()
        self.hidden_dim = hidden_dim
        
        # The LSTM takes data inputs and outputs hidden states
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to pitch_type space
        self.linear = nn.Linear(hidden_dim, num_pitches)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, inputs):
        inputs = inputs.view(len(inputs), 1, -1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        pitch_type_space = self.linear(lstm_out.view(len(inputs), -1))
        pitch_type_scores = F.softmax(pitch_type_space, dim=1)
        
        return pitch_type_scores


N_EPOCHS = 10
HIDDEN_DIM = 128

df = pd.read_csv('lestj001-2016.csv')
df = pd.concat([pd.get_dummies(df.pitch_type), df], axis=1)
df = df.dropna()

cuda0 = torch.device('cuda:0')

unique_pitches = pd.get_dummies(df.pitch_type).columns

def prepare_data(df, target=False):
    if target:
        return torch.tensor(np.array(df[unique_pitches]),
                            dtype=torch.float32)
    else:
        return torch.tensor(np.array(df[[
            'previous_o','previous_s', 'previous_b',
            'away_team_runs','home_team_runs','num',
            'inning','temp','windspeed',
            'previous_start_speed','previous_spin_rate',
            'previous_zone',
            'previous_nasty',
        ]]),
                          dtype=torch.float32)

data = [df[df['ab_num']==ab_num] for ab_num in df.ab_num.unique()]

split_pct = 0.80 
train_data = data[:int(len(data)*split_pct)]
test_data = data[int(len(data)*split_pct):]

#train, test = input_data.split(int(input_data.size()[0]*split_pct))
#out_train, out_test = input_data.split(int(input_data.size()[0]*split_pct))
full_input = prepare_data(df)
full_output = prepare_data(df, True)
model = Sequence(HIDDEN_DIM, full_input.size()[1], full_output.size()[1])
loss_function = nn.MultiLabelSoftMarginLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_data(data[0])
    pitch_type_scores = model(inputs)
    print(pitch_type_scores)


for epoch in range(N_EPOCHS): 
    for dataframe in train_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        inputs = prepare_data(dataframe)
        pitch_types = prepare_data(dataframe,True)

        # Step 3. Run our forward pass.
        pitch_type_scores = model(inputs)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(pitch_type_scores, pitch_types)
        loss.backward()
        optimizer.step()

#See what the scores are after training
with torch.no_grad():
    inputs = prepare_data(test_data[1])
    pitch_type_scores = model(inputs)
    pitch_types = prepare_data(test_data[1],True)
    print(pitch_type_scores)

    print(pitch_types)
