import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 用GPU训练，没有就用CPU
print(f"Using device: {device}")

## 加载数据
loadFile = 'rand_data.mat'
loaddata = scipy.io.loadmat(loadFile)
X = loaddata['X']  # 风格1训练小批量数据，尺寸为[帧数,批次大小,维度]
Y = loaddata['Y']  # 风格2训练小批量数据，尺寸为[帧数,批次大小,维度]
x = loaddata['feats_x']  # 风格1测试数据，尺寸为[帧数,维度]
y = loaddata['feats_y']  # 风格2测试数据，尺寸为[帧数,维度]

## PARAMETERS
residual_channels = 256
filter_width = 11
dilations = [1, 1, 1, 1, 1, 1]
input_channels = X[0][0].shape[2]
output_channels = X[0][0].shape[2]
cond_dim = None
postnet_channels = 256
do_postproc = True
do_gu = True

# 优化器参数
adam_lr = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999
num_epochs = 5
n_critic = 5


class CNET(nn.Module):
    def __init__(self,
                 name,
                 residual_channels=64,
                 filter_width=3,
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 input_channels=123,
                 output_channels=48,
                 cond_dim=None,
                 cond_channels=64,
                 postnet_channels=256,
                 do_postproc=True,
                 do_GU=True):

        super(CNET, self).__init__()

        self.name = name
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.postnet_channels = postnet_channels
        self.do_postproc = do_postproc
        self.do_GU = do_GU
        self._use_cond = cond_dim is not None

        if self._use_cond:
            self.cond_dim = cond_dim
            self.cond_channels = cond_channels

        self._create_layers()

    def _create_layers(self):
        fw = self.filter_width
        r = self.residual_channels
        s = self.postnet_channels

        # Input layer
        self.input_conv = nn.Conv1d(
            self.input_channels, 2 * r,
            kernel_size=fw, padding=(fw - 1) // 2
        )

        # Conditional embedding
        if self._use_cond:
            self.cond_embed = nn.Conv1d(
                self.cond_dim, self.cond_channels,
                kernel_size=1, padding=0
            )

        # Convolution modules
        self.conv_modules = nn.ModuleList()
        for i, dilation in enumerate(self.dilations):
            module = nn.ModuleDict()

            # Filter and gate convolution
            padding = (dilation * (fw - 1)) // 2
            module['filter_gate'] = nn.Conv1d(
                r, 2 * r,
                kernel_size=fw,
                dilation=dilation,
                padding=padding
            )

            # Skip connection
            if self.do_postproc:
                module['skip'] = nn.Conv1d(r, s, kernel_size=1)

            # Gated unit
            if self.do_GU:
                module['post_filter'] = nn.Conv1d(r, r, kernel_size=1)

            # Conditional convolution
            if self._use_cond:
                module['cond_filter'] = nn.Conv1d(
                    self.cond_channels, 2 * r, kernel_size=1
                )

            self.conv_modules.append(module)

        # Post-processing module
        if self.do_postproc:
            self.postproc_conv1 = nn.Conv1d(
                s, s, kernel_size=fw, padding=(fw - 1) // 2
            )

            if isinstance(self.output_channels, list):
                output_dim = sum(self.output_channels)
            else:
                output_dim = self.output_channels

            self.postproc_conv2 = nn.Conv1d(
                s, output_dim, kernel_size=fw, padding=(fw - 1) // 2
            )

            # 添加ReLU激活函数
            self.relu = nn.ReLU()
        else:
            # Last layer
            if isinstance(self.output_channels, list):
                output_dim = sum(self.output_channels)
            else:
                output_dim = self.output_channels

            self.last_conv = nn.Conv1d(
                r, output_dim, kernel_size=fw, padding=(fw - 1) // 2
            )

    def _input_layer(self, x):
        # x shape: (batch, seq_len, input_channels)
        x = x.transpose(1, 2)  # (batch, input_channels, seq_len)

        y = self.input_conv(x)

        # Split for tanh and sigmoid gates
        r = self.residual_channels
        y_tanh = torch.tanh(y[:, :r, :])
        y_sigmoid = torch.sigmoid(y[:, r:, :])
        y = y_tanh * y_sigmoid

        return y  # (batch, residual_channels, seq_len)

    def _embed_cond(self, cond_input):
        # cond_input shape: (batch, seq_len, cond_dim)
        cond_input = cond_input.transpose(1, 2)  # (batch, cond_dim, seq_len)

        y = self.cond_embed(cond_input)
        y = torch.tanh(y)

        return y  # (batch, cond_channels, seq_len)

    def _conv_module(self, x, module, dilation, cond_input=None):
        # x shape: (batch, residual_channels, seq_len)

        # Filter and gate convolution
        y = module['filter_gate'](x)

        # Add conditional input if available
        if self._use_cond and cond_input is not None:
            cond_contribution = module['cond_filter'](cond_input)
            y = y + cond_contribution

        # Split for tanh and sigmoid gates
        r = self.residual_channels
        y_tanh = torch.tanh(y[:, :r, :])
        y_sigmoid = torch.sigmoid(y[:, r:, :])
        y = y_tanh * y_sigmoid

        # Skip connection
        if self.do_postproc:
            skip_out = module['skip'](y)
        else:
            skip_out = None

        # Gated unit
        if self.do_GU:
            y = module['post_filter'](y)
            y = y + x  # Residual connection

        return y, skip_out

    def _postproc_module(self, skip_outputs):
        # Sum all skip outputs
        x = torch.stack(skip_outputs, dim=0).sum(dim=0)

        y = self.postproc_conv1(x)
        y = self.relu(y)  # 使用类属性而不是F.relu
        y = self.postproc_conv2(y)

        # Split output if needed
        if isinstance(self.output_channels, list):
            outputs = []
            start = 0
            for channels in self.output_channels:
                outputs.append(y[:, start:start + channels, :])
                start += channels
            return outputs
        else:
            return y

    def _last_layer(self, x):
        y = self.last_conv(x)

        # Split output if needed
        if isinstance(self.output_channels, list):
            outputs = []
            start = 0
            for channels in self.output_channels:
                outputs.append(y[:, start:start + channels, :])
                start += channels
            return outputs
        else:
            return y

    def forward(self, x_input, cond_input=None):
        # x_input shape: (batch, seq_len, input_channels)
        batch_size, seq_len, input_channels = x_input.shape

        # Embed conditional input if available
        if self._use_cond and cond_input is not None:
            cond_embedded = self._embed_cond(cond_input)
        else:
            cond_embedded = None

        # Input layer
        x = self._input_layer(x_input)

        # Convolution modules
        skip_outputs = []
        for i, (module, dilation) in enumerate(zip(self.conv_modules, self.dilations)):
            x, skip_out = self._conv_module(x, module, dilation, cond_embedded)
            if self.do_postproc and skip_out is not None:
                skip_outputs.append(skip_out)

        # Output processing
        if self.do_postproc:
            output = self._postproc_module(skip_outputs)
        else:
            output = self._last_layer(x)

        # Convert back to (batch, seq_len, channels) format
        if isinstance(output, list):
            output = [out.transpose(1, 2).contiguous() for out in output]
        else:
            output = output.transpose(1, 2).contiguous()

        return output

    def get_variable_list(self):
        """返回模型参数列表（兼容性函数）"""
        return list(self.parameters())


# 初始化模型
G = CNET(name='G',
         input_channels=input_channels,
         output_channels=output_channels,
         residual_channels=residual_channels,
         filter_width=filter_width,
         dilations=dilations,
         postnet_channels=postnet_channels,
         cond_dim=cond_dim,
         do_postproc=do_postproc,
         do_GU=do_gu).to(device)

F = CNET(name='F',
         input_channels=input_channels,
         output_channels=output_channels,
         residual_channels=residual_channels,
         filter_width=filter_width,
         dilations=dilations,
         postnet_channels=postnet_channels,
         cond_dim=cond_dim,
         do_postproc=do_postproc,
         do_GU=do_gu).to(device)

D_x = CNET(name='D_x',
           input_channels=output_channels,
           output_channels=1,
           residual_channels=residual_channels,
           filter_width=filter_width,
           dilations=dilations,
           postnet_channels=postnet_channels,
           cond_dim=cond_dim,
           do_postproc=do_postproc,
           do_GU=do_gu).to(device)

D_y = CNET(name='D_y',
           input_channels=output_channels,
           output_channels=1,
           residual_channels=residual_channels,
           filter_width=filter_width,
           dilations=dilations,
           postnet_channels=postnet_channels,
           cond_dim=cond_dim,
           do_postproc=do_postproc,
           do_GU=do_gu).to(device)

# 优化器
gen_optimizer = optim.Adam(list(G.parameters()) + list(F.parameters()),
                           lr=adam_lr, betas=(adam_beta1, adam_beta2))
d_optimizer = optim.Adam(list(D_x.parameters()) + list(D_y.parameters()),
                         lr=adam_lr, betas=(adam_beta1, adam_beta2))

# 训练记录
lossD_all = np.zeros((num_epochs * X.shape[0], 7), dtype=float)
lossG_all = np.zeros((num_epochs * X.shape[0], 7), dtype=float)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """计算梯度惩罚"""
    batch_size, seq_len, channels = real_samples.shape

    # 创建插值点
    alpha = torch.rand(batch_size, seq_len, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = interpolates.requires_grad_(True)

    # 判别器输出
    d_interpolates = D(interpolates)

    # 计算梯度
    grad_outputs = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # 确保梯度是连续的
    gradients = gradients.contiguous()

    # 重塑梯度张量并计算惩罚
    gradients = gradients.reshape(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    return gradient_penalty


# 训练循环
cont = 0
for epoch in range(num_epochs):
    # 随机排列数据
    idx = np.random.permutation(X.shape[0])

    for batch_i in range(X.shape[0]):
        # 准备数据
        x_real_batch = torch.FloatTensor(X[idx[batch_i]][0]).to(device)
        y_real_batch = torch.FloatTensor(Y[idx[batch_i]][0]).to(device)

        # 训练判别器
        for critic_i in range(n_critic):
            d_optimizer.zero_grad()

            # 前向传播
            y_hat = G(x_real_batch)
            x_hat = F(y_real_batch)

            # 判别器输出
            D_out_y_real = D_y(y_real_batch)
            D_out_y_fake = D_y(y_hat.detach())
            D_out_x_real = D_x(x_real_batch)
            D_out_x_fake = D_x(x_hat.detach())

            # WGAN损失
            D_y_loss_gan = -torch.mean(D_out_y_real) + torch.mean(D_out_y_fake)
            D_x_loss_gan = -torch.mean(D_out_x_real) + torch.mean(D_out_x_fake)

            # 梯度惩罚
            gradient_penalty_y = compute_gradient_penalty(D_y, y_real_batch, y_hat.detach())
            gradient_penalty_x = compute_gradient_penalty(D_x, x_real_batch, x_hat.detach())

            # 零惩罚
            D_loss_zeropen_x = 1e-2 * torch.sum(torch.square(D_out_x_real))
            D_loss_zeropen_y = 1e-2 * torch.sum(torch.square(D_out_y_real))

            # 总判别器损失
            D_loss = (D_y_loss_gan + D_x_loss_gan +
                      10 * gradient_penalty_y + 10 * gradient_penalty_x +
                      D_loss_zeropen_x + D_loss_zeropen_y)

            D_loss.backward()
            d_optimizer.step()

            # 记录损失
            if critic_i == n_critic - 1:
                lossD_all[cont][0] = D_loss.item()
                lossD_all[cont][1] = D_x_loss_gan.item()
                lossD_all[cont][2] = D_y_loss_gan.item()
                lossD_all[cont][3] = gradient_penalty_y.item()
                lossD_all[cont][4] = gradient_penalty_x.item()
                lossD_all[cont][5] = D_loss_zeropen_x.item()
                lossD_all[cont][6] = D_loss_zeropen_y.item()

        # 训练生成器
        gen_optimizer.zero_grad()

        # 前向传播
        y_hat = G(x_real_batch)
        x_hat_hat = F(y_hat)
        x_id = F(x_real_batch)

        x_hat = F(y_real_batch)
        y_hat_hat = G(x_hat)
        y_id = G(y_real_batch)

        # 判别器输出
        D_out_y_fake = D_y(y_hat)
        D_out_x_fake = D_x(x_hat)

        # GAN损失
        G_loss_gan = -torch.mean(D_out_y_fake)
        F_loss_gan = -torch.mean(D_out_x_fake)

        # 重建损失
        recon_loss_x = 10 * torch.mean(torch.abs(x_real_batch - x_hat_hat))
        recon_loss_y = 10 * torch.mean(torch.abs(y_real_batch - y_hat_hat))

        # 身份损失
        id_loss_x = 5 * torch.mean(torch.abs(x_real_batch - x_id))
        id_loss_y = 5 * torch.mean(torch.abs(y_real_batch - y_id))

        # 总生成器损失
        if epoch < 50:
            Gen_loss = (G_loss_gan + F_loss_gan + recon_loss_x + recon_loss_y +
                        id_loss_x + id_loss_y)
        else:
            Gen_loss = G_loss_gan + F_loss_gan + recon_loss_x + recon_loss_y

        Gen_loss.backward()
        gen_optimizer.step()

        # 记录生成器损失
        lossG_all[cont][0] = Gen_loss.item()
        lossG_all[cont][1] = G_loss_gan.item()
        lossG_all[cont][2] = F_loss_gan.item()
        lossG_all[cont][3] = recon_loss_x.item()
        lossG_all[cont][4] = recon_loss_y.item()
        lossG_all[cont][5] = id_loss_x.item() if epoch < 50 else 0
        lossG_all[cont][6] = id_loss_y.item() if epoch < 50 else 0

        cont += 1

        print(f"Epoch {epoch}, Batch {batch_i}: Gen loss {Gen_loss.item():.4f}, D loss {D_loss.item():.4f}")

    print(f"Epoch {epoch}: Gen loss {Gen_loss.item():.4f}, D loss {D_loss.item():.4f}")

    # 保存损失
    saveFile2 = './errors.mat'
    scipy.io.savemat(saveFile2, {"lossG_all": lossG_all, "lossD_all": lossD_all})

# 测试
saveFile1 = './pred_res.mat'

# 测试x
no_utt = x.shape[0]
y_pred = np.ndarray((no_utt,), dtype=object)
x_recon = np.ndarray((no_utt,), dtype=object)
x_pred_id = np.ndarray((no_utt,), dtype=object)

G.eval()
F.eval()
with torch.no_grad():
    for n_val in range(no_utt):
        input_data = np.reshape(x[n_val][0], (1, x[n_val][0].shape[0], x[n_val][0].shape[1]))
        if input_data.shape[0] != 0:
            input_tensor = torch.FloatTensor(input_data).to(device)
            y_pred[n_val] = G(input_tensor).cpu().numpy()
            x_recon[n_val] = F(torch.FloatTensor(y_pred[n_val]).to(device)).cpu().numpy()
            x_pred_id[n_val] = F(input_tensor).cpu().numpy()
        else:
            y_pred[n_val] = np.nan
            x_recon[n_val] = np.nan
            x_pred_id[n_val] = np.nan

    # 测试y
    no_utt = y.shape[0]
    x_pred = np.ndarray((no_utt,), dtype=object)
    y_recon = np.ndarray((no_utt,), dtype=object)
    y_pred_id = np.ndarray((no_utt,), dtype=object)

    for n_val in range(no_utt):
        input_data = np.reshape(y[n_val][0], (1, y[n_val][0].shape[0], y[n_val][0].shape[1]))
        if input_data.shape[0] != 0:
            input_tensor = torch.FloatTensor(input_data).to(device)
            x_pred[n_val] = F(input_tensor).cpu().numpy()
            y_recon[n_val] = G(torch.FloatTensor(x_pred[n_val]).to(device)).cpu().numpy()
            y_pred_id[n_val] = G(input_tensor).cpu().numpy()
        else:
            x_pred[n_val] = np.nan
            y_recon[n_val] = np.nan
            y_pred_id[n_val] = np.nan

# 保存结果
scipy.io.savemat(saveFile1, {
    "y_pred": y_pred,
    "x_recon": x_recon,
    "x_pred_id": x_pred_id,
    "x_pred": x_pred,
    "y_recon": y_recon,
    "y_pred_id": y_pred_id
})

# 保存模型
torch.save({
    'G_state_dict': G.state_dict(),
    'F_state_dict': F.state_dict(),
    'D_x_state_dict': D_x.state_dict(),
    'D_y_state_dict': D_y.state_dict(),
}, './models.pth')

print("Training completed and results saved!")