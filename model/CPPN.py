import torch
import torch.nn as nn
import numpy as np
import pdb

class CPPN(nn.Module):
    """
    A CPPN model, mapping a number of input coordinates to a multidimensional output (e.g. color)
    """
    def __init__(self, model_definition: dict) -> None:
        """
        Args:
            model_definition: dictionary containing all the needed parameters
                - num_layers: number of hidden layers
                - num_filters: number of filters in the hidden blocks
                - num_input_channels: number of expected input channels
                - num_input_channels_views: number of expected input channels for the view direction
                - num_output_channels: number of expected output channels
                - use_bias: whether biases are used
                - pos_enc: which positional encoding to apply: 'none', 'fourier', 'barf'
                - pos_enc_basis: basis for positional encoding (L)
                - pos_enc_basis_views: basis for positional encoding for views (L)
                - num_img: number of images for training (translation/rotation)
        """
        super().__init__()
        self.version = "v0.00"
        self.model_definition = model_definition
        self.device = model_definition['device']

        # getting the parameters
        self.num_early_layers = model_definition['num_early_layers']
        self.num_late_layers = model_definition['num_late_layers']
        self.num_filters = model_definition['num_filters']
        self.num_input_channels = model_definition['num_input_channels'] # x,y,z
        self.num_input_channels_views = model_definition['num_input_channels_views'] # direction unit vector
        self.num_output_channels = model_definition['num_output_channels']
        self.use_bias = model_definition['use_bias']
        self.use_pos_enc = model_definition['pos_enc']
        self.act_func = model_definition['act_func']
        self.num_img = model_definition['num_img']

        self.mult_img = self.num_img > 1
        self.use_viewdirs = self.num_input_channels_views > 0
        self.enc_fun = None

        input_features = self.num_input_channels
        input_features_views = self.num_input_channels_views
        num_filters = self.num_filters
        use_bias = self.use_bias
        num_output_channels = self.num_output_channels

        # SIREN
        self.first_act_func = nn.ReLU()
        self.act_func = nn.ReLU()
        if model_definition['act_func'] == 'sine':
            self.first_act_func = Sine(w0=model_definition['sine_weights'])
            self.act_func = Sine()
        elif model_definition['act_func'] == 'tanh':
            self.first_act_func = nn.Tanh()
            self.act_func = nn.Tanh()

        if self.use_pos_enc != 'none':
            self.pos_enc_basis = model_definition['pos_enc_basis']
            input_features = self.num_input_channels + self.num_input_channels * 2 * self.pos_enc_basis

            if self.use_viewdirs:
                self.pos_enc_basis_views = model_definition['pos_enc_basis_views']
                input_features_views = self.num_input_channels_views + self.num_input_channels_views * 2 * self.pos_enc_basis_views
            
            if self.use_pos_enc == 'fourier' and 'fourier_sigma' in model_definition:
                self.enc_fun = self.fourier_pos_enc
                self.fourier_sigma = model_definition['fourier_sigma']
                self.fourier_coefficients = nn.Parameter(
                    torch.randn([self.num_input_channels * self.pos_enc_basis]) *
                    self.fourier_sigma)

                if self.use_viewdirs:
                    self.fourier_coefficients_views = nn.Parameter(
                    torch.randn([self.num_input_channels_views * self.pos_enc_basis_views]) *
                    self.fourier_sigma)
                
            if self.use_pos_enc == 'barf':
                self.enc_fun = self.barf_pos_enc
                self.k_values = torch.repeat_interleave(torch.arange(0., self.pos_enc_basis), self.num_input_channels)
                self.barf_freq = torch.Tensor(2**self.k_values*np.pi).to(self.device)

                # TODO: view pos encoding
                if self.use_viewdirs:
                    self.k_values_views = torch.repeat_interleave(torch.arange(0., self.pos_enc_basis_views), self.num_input_channels_views)
                    self.barf_freq_views = torch.Tensor(2**self.k_values_views*np.pi).to(self.device)
                    self.update_barf_alpha(0, 'views')

                # start with alpha = 0
                self.update_barf_alpha(0, 'pts')

        # creating the learnable blocks
        early_pts_layers = []
        # input layer
        early_pts_layers += self.__create_layer(input_features, num_filters,
                                           use_bias, activation=self.first_act_func)
        # hidden layers: early
        for _ in range(self.num_early_layers):
            early_pts_layers += self.__create_layer(num_filters, num_filters,
                                               use_bias, activation=self.act_func)

        self.early_pts_layers = nn.ModuleList(early_pts_layers)

        # skip connection
        if self.num_late_layers > 0:
            self.skip_connection = self.__create_layer(num_filters + input_features, num_filters,
                                                use_bias, activation=self.act_func)

            late_pts_layers = []
            for _ in range(self.num_late_layers - 1):
                late_pts_layers += self.__create_layer(num_filters, num_filters,
                                                use_bias, activation=self.act_func)

            self.late_pts_layers = nn.ModuleList(late_pts_layers)

        if self.use_viewdirs:
            self.views_layers = self.__create_layer(input_features_views + num_filters, num_filters//2,
                                        use_bias, activation=self.act_func)
            self.alpha_linear = self.__create_layer(num_filters, num_output_channels - 1,
                                        use_bias, activation=None)
            self.feature_linear = nn.Linear(num_filters, num_filters)
            self.rgb_linear = self.__create_layer(num_filters//2, num_output_channels - 1,
                                        use_bias, activation=None)
        # output layer
        else:
            self.output_linear = self.__create_layer(num_filters, num_output_channels,
                                        use_bias, activation=None)
        
        # translation
        self.img1 = nn.Parameter(torch.tensor([0., 0.], dtype=torch.float))
        self.img2 = nn.Parameter(torch.tensor([0., 0.], dtype=torch.float))
        
        # model understanding API
        self.store_activations = False
        self.activation_dictionary = {}

    @staticmethod
    def __create_layer(num_in_filters: int, num_out_filters: int,
                       use_bias: bool, activation=nn.ReLU(), dropout=0.5) -> nn.Sequential:
        block = []
        block.append(nn.Linear(num_in_filters, num_out_filters, bias=use_bias)) # Dense layer
        if activation:
            block.append(activation)
            # block.append(nn.Dropout(dropout))
        block = nn.Sequential(*block)

        return block

    def activations(self, store_activations: bool) -> None:
        """
        Configure the model to retain or discard the activations during the forward pass

        Args:
            activations (bool): keep/discard the activations during inference
        """

        self.store_activations = store_activations

        if not store_activations:
            self.activation_dictionary = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input_pts, input_views = torch.split(x, [self.num_input_channels, self.num_input_channels_views], dim=-1)
        
        values = input_pts
        views = input_views

        # positional encoding
        pos_enc = self.use_pos_enc
        pts_encoded = input_pts
        views_encoded = input_views
        if pos_enc != 'none':
            pts_encoded = self.pos_enc(input_pts, self.pos_enc_basis, 'pts')
            if self.use_viewdirs:
                views_encoded = self.pos_enc(views, self.pos_enc_basis_views, 'views')

        values = pts_encoded
        for _, pts_layer in enumerate(self.early_pts_layers):
            values = pts_layer(values)

        if self.num_late_layers > 0:
            values = self.skip_connection(torch.cat([pts_encoded, values], dim=-1))

            for _, pts_layer in enumerate(self.late_pts_layers):
                values = pts_layer(values)
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(values)
            feature = self.feature_linear(values)
            values = torch.cat([feature, views_encoded], -1)

            for layer in self.views_layers:
                values = layer(values)
            
            rgb = self.rgb_linear(values)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(values)

        return outputs

    def pos_enc(self, values, pos_enc_basis, type):
        input_values = values
        if pos_enc_basis > 0:
            basis_values = torch.cat(pos_enc_basis * [input_values], dim=-1)
            pos_values = torch.cat(self.enc_fun(basis_values, type), dim=-1)
            fin_values = torch.cat([input_values, pos_values], dim=-1)
        else: fin_values = input_values
        return fin_values

    def fourier_pos_enc(self, values, type):
        coefficients = self.fourier_coefficients
        if type == 'views':
            coefficients = self.fourier_coefficients_views

        value = 2 * np.pi * values * coefficients
        return [torch.sin(value),  torch.cos(value)]

    def barf_pos_enc(self, values, type):
        # pos enc
        if type == 'views':
            value = self.barf_freq_views*values
            weights = self.barf_weights_views
        else:
            value = self.barf_freq*values
            weights = self.barf_weights

        weights = weights.to(self.device)
        return [weights * torch.sin(value), weights * torch.cos(value)]

    def update_barf_alpha(self, barf_alpha, type):
        if type == 'views':
            self.barf_alpha_views = barf_alpha
            self.barf_weights_views = self.barf_coefficients(barf_alpha, self.k_values_views)
        else:
            self.barf_alpha = barf_alpha
            self.barf_weights = self.barf_coefficients(barf_alpha, self.k_values)

    def barf_coefficients(self, barf_alpha, k_values):
        weights = []
        for k in k_values:
            barf_k = barf_alpha - (k + 1)
            new_val = 0
            if barf_k < 0: 
                new_val = 0
            elif barf_k >= 0 and barf_k < 1:
                new_val = (1 - torch.cos((barf_alpha - k + 1)* 3.1415))/2
            elif barf_k >= 1:
                new_val = 1
            weights.append(new_val)

        weights = nn.Parameter(torch.Tensor(weights))

        return weights

    def save(self, filename: str, training_information: dict) -> None:
        """
        Save the CPPN model

        Args:
            filename (str): path filepath on which the model will be saved
            training_information (dict): dictionary containing information on the training
        """
        torch.save(
            {
                'version': self.version,
                'parameters': self.model_definition,
                'training_information': training_information,
                'model': self.state_dict(),
            },
            f=filename)

class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.
        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')