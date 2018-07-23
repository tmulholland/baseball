import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        pitch_type_space = self.linear(lstm_out.view(len(inputs), -1))
        pitch_type_scores = F.log_softmax(pitch_type_space, dim=1)
        
        return pitch_type_scores


N_EPOCHS = 10
HIDDEN_DIM = 64

model = Sequence(HIDDEN_DIM, len(train), len(test))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0])
    pitch_type_scores = model(inputs)
    print(pitch_type_scores)

for epoch in range(n_epochs): 
    for inputs, pitch_types in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        inputs = prepare_sequence(input_data)
        targets = prepare_sequence(pitch_type_training_data)

        # Step 3. Run our forward pass.
        pitch_type_scores = model(inputs)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(pitch_type_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    pitch_type_scores = model(inputs)

    print(pitch_type_scores)
