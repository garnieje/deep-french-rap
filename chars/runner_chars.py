from lstm_char import LstmChars
from data_reader import DataReader
import torch.nn as nn
import torch
from torch.autograd import Variable

import time
import math

PATH_LYRICS = "/data/data/lyrics_Iam.csv"
# create the folder to store the models
PATH_SAVE = "./models/"

reader = DataReader(PATH_LYRICS)
ix_to_char = reader.get_ix_to_char()
char_to_ix = reader.get_char_to_ix()
n_characters = len(ix_to_char)

n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 8
n_layers = 1
lr = 0.005
seq_len = 20
batch_size = 2
criterion = nn.CrossEntropyLoss()

cuda = False


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

decoder = LstmChars(n_characters, hidden_size, n_characters,
                    n_layers, lr)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
loss_avg = 0
all_losses = []
start = time.time()

def char_tensor(string):

    tensor = torch.zeros(len(string)).long()
    for i in range(len(string)):
        try:
            tensor[i] = char_to_ix[string[i]]
        except Exception as e:
            continue
    return tensor

def train(inp, targets, batch_size):

    hidden = decoder.init_hidden(batch_size)
    if cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(inp.size(1)):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), targets[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / inp.size(1)

def evaluate(prime="Le", predict_len=50, temperature=0.8):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()

    predicted = prime

    for p in range(len(prime) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = ix_to_char[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted

def save(path, epoch):
    save_filename = path + 'model_char_' + str(epoch) + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


for epoch in range(1, n_epochs + 1):

    inputs, targets = reader.get_batch(batch_size, seq_len)
    if cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
    loss = train(inputs, targets, batch_size)
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate())
        save(PATH_SAVE, epoch)



    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0
