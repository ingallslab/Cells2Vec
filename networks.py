import torch
import xgboost as xgb
# CausalCNNEncoder forked from https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/tree/master


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """

    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """

    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2 
             # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """

    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )

    def forward(self, x):
        return self.network(x)

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, latent_dim, dropout=0.0):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
       
        self.latent_dim = latent_dim
        self.num_layers = len(input_size)
        
        self.dropout = dropout

        self.lstm_layers = torch.nn.ModuleList()
        for i in range(self.num_layers):

            in_size = self.input_size[0] if i == 0 else self.input_size[i-1]
            self.lstm_layers.append(
                torch.nn.LSTM(input_size=in_size, hidden_size=self.input_size[i], batch_first=True))

        self.reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        self.squeeze = SqueezeChannels()

        self.linear = torch.nn.Linear(self.input_size[-1], self.latent_dim)

    def forward(self, x):
        for i in range(self.num_layers):
            x, _ = self.lstm_layers[i](x)
        
        x = self.reduce_size(x.transpose(1, 2))
     
        x = self.squeeze(x)
        
        x = self.linear(x)

        return x

class StackedLSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size,dropout_rate=0.2,pooling='max'):
        super(StackedLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.dropout=torch.nn.Dropout(self.dropout_rate)
        self.pooling = pooling

        self.lstm_layers = torch.nn.ModuleList([torch.nn.LSTM(input_size, hidden_size, batch_first=True)])

        for _ in range(1, num_layers):
            self.lstm_layers.append(torch.nn.LSTM(hidden_size, hidden_size, batch_first=True))

        if pooling not in ['max', 'avg']:
            raise ValueError("Invalid pooling type. Use 'max' or 'avg'.")

        if self.pooling == 'max':
            self.pool = torch.nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = torch.nn.AdaptiveAvgPool1d(1)

        self.linear_layer = torch.nn.Linear(hidden_size, embedding_size)

    def forward(self, x):

        h_states = []
        c_states = []
        for _ in range(self.num_layers):
            h_states.append(torch.zeros(x.size(0), self.hidden_size).to(x.device))
            c_states.append(torch.zeros(x.size(0), self.hidden_size).to(x.device))

     
        for i, lstm in enumerate(self.lstm_layers):
            x, (h_states[i], c_states[i]) = lstm(x, (h_states[i].unsqueeze(0), c_states[i].unsqueeze(0)))
            if i < self.num_layers:
                x= self.dropout(x)

        x = x.permute(0, 2, 1)  # Convert (batch_size, sequence_length, hidden_size) to (batch_size, hidden_size, sequence_length)
        x = self.pool(x).squeeze(2) # Output shape: (batch_size, hidden_size)

        # Final Linear layer to get the embedding
        embedding = self.linear_layer(x)

        return embedding


class StackedGRUEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size,dropout_rate=0.2, pooling='max'):
        super(StackedGRUEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.dropout=torch.nn.Dropout(self.dropout_rate)
        self.pooling = pooling

        self.gru_layers = torch.nn.ModuleList(
            [torch.nn.GRU(input_size, hidden_size, batch_first=True)])

        for _ in range(1, num_layers):
            self.gru_layers.append(torch.nn.GRU(
                hidden_size, hidden_size, batch_first=True))

        if pooling not in ['max', 'avg']:
            raise ValueError("Invalid pooling type. Use 'max' or 'avg'.")

        if self.pooling == 'max':
            self.pool = torch.nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = torch.nn.AdaptiveAvgPool1d(1)

        self.linear_layer = torch.nn.Linear(hidden_size, embedding_size)

    def forward(self, x):

        h_states = []
        for _ in range(self.num_layers):
            h_states.append(torch.zeros(
                x.size(0), self.hidden_size).to(x.device))

        for i, gru in enumerate(self.gru_layers):
            x, h_states[i] = gru(x, h_states[i].unsqueeze(0))
            if i < self.num_layers:
                x= self.dropout(x)


        # Convert (batch_size, sequence_length, hidden_size) to (batch_size, hidden_size, sequence_length)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(2)  # Output shape: (batch_size, hidden_size)

        # Final Linear layer to get the embedding
        embedding = self.linear_layer(x)

        return embedding
    

class Tripletnet(torch.nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        
        return embedded_x, embedded_y, embedded_z
    