import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
import math

# =============================================================================
# Lightweight attention modules
# =============================================================================

class ECAAttention(nn.Module):
    """Efficient Channel Attention (ECA)"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).view(b, 1, c)
        y = self.conv(y).view(b, c)
        y = self.sigmoid(y)
        return x * y.unsqueeze(-1) if len(x.shape) == 3 else x * y

class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention"""
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.size()
        y = torch.mean(x, dim=-1) if len(x.shape) == 3 else x
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.unsqueeze(-1) if len(x.shape) == 3 else x * y

class SimAM(nn.Module):
    """Simple Attention Module - parameter-free attention"""
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c = x.size()[:2]
        if len(x.shape) == 3:
            n = x.size(2)
            x_minus_mu_square = (x - x.mean(dim=-1, keepdim=True)).pow(2)
            y = x_minus_mu_square / (4 * (x_minus_mu_square.mean(dim=-1, keepdim=True) + self.e_lambda)) + 0.5
        else:
            x_minus_mu_square = (x - x.mean(dim=-1, keepdim=True)).pow(2)
            y = x_minus_mu_square / (4 * (x_minus_mu_square.mean(dim=-1, keepdim=True) + self.e_lambda)) + 0.5
        return x * self.activaton(y)

class LinearAttention(nn.Module):
    """Linear Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.eps = eps 
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Use ReLU instead of ELU to avoid numerical instability
        q = F.relu(q) + self.eps
        k = F.relu(k) + self.eps

        # Linear attention computation
        k_cumsum = k.sum(dim=-2, keepdim=True)
        
        # Safe reciprocal computation
        denominator = (q * k_cumsum).sum(dim=-1, keepdim=True)
        denominator = torch.clamp(denominator, min=self.eps)
        D_inv = 1. / denominator
        
        # Clamp D_inv to prevent numerical explosion
        D_inv = torch.clamp(D_inv, max=1e6)
        
        context = k.transpose(-2, -1) @ v
        attn = q @ context
        attn = attn * D_inv
        
        attn = attn.transpose(1, 2).reshape(B, N, C)
        attn = self.proj(attn)
        return attn

class StableLocalGlobalAttention(nn.Module):
    """Local-Global Attention"""
    def __init__(self, dim, num_heads=8, window_size=16, qkv_bias=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.eps = eps
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Local attention
        self.local_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.local_proj = nn.Linear(dim, dim // 2)
        
        # Global attention (linear attention)
        self.global_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.global_proj = nn.Linear(dim, dim // 2)
        
        # Fusion layer
        self.fusion = nn.Linear(dim, dim)
        
        # Position encoding for local relation modeling
        self.pos_encoding = nn.Sequential(
            nn.Linear(3, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4)
        )

    def forward(self, x, coords=None):
        """
        Args:
            x: [B, N, C] features
            coords: [B, N, 3] optional coordinates for spatial local attention
        """
        B, N, C = x.shape
        
        # Improved local attention: sliding window
        if N > self.window_size:
            # Split sequence into overlapping windows
            stride = max(1, self.window_size // 2)  # 50% overlap
            num_windows = (N - self.window_size) // stride + 1
            
            local_outputs = []
            
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = min(start_idx + self.window_size, N)
                if end_idx - start_idx < self.window_size // 2:
                    break
                    
                # Extract current window
                window_size_curr = end_idx - start_idx
                x_window = x[:, start_idx:end_idx]  # [B, window_size_curr, C]
                
                # Generate QKV
                window_qkv = self.local_qkv(x_window).reshape(
                    B, window_size_curr, 3, self.num_heads, C // self.num_heads)
                window_qkv = window_qkv.permute(2, 0, 3, 1, 4)
                window_q, window_k, window_v = window_qkv.unbind(0)
                
                # Add position encoding if coordinates available
                if coords is not None:
                    window_coords = coords[:, start_idx:end_idx]
                    # Compute relative positions
                    rel_pos = window_coords.unsqueeze(2) - window_coords.unsqueeze(1)  # [B, N, N, 3]
                    pos_bias = self.pos_encoding(rel_pos)  # [B, N, N, C//4]
                    # Add position bias to attention scores
                    pos_bias = pos_bias.mean(dim=-1).unsqueeze(1)  # [B, 1, N, N]
                else:
                    pos_bias = 0
                
                # Standard attention computation
                window_attn = (window_q @ window_k.transpose(-2, -1)) * self.scale + pos_bias
                window_attn = F.softmax(window_attn, dim=-1)
                
                window_out = (window_attn @ window_v).transpose(1, 2).reshape(
                    B, window_size_curr, C)
                window_local_out = self.local_proj(window_out)
                
                local_outputs.append(window_local_out)
            
            # Smart merge of overlapping window outputs
            if len(local_outputs) > 1:
                # Reconstruct full sequence, handle overlaps
                local_out = torch.zeros(B, N, C // 2, device=x.device)
                overlap_count = torch.zeros(N, device=x.device)
                
                for i, window_out in enumerate(local_outputs):
                    start_idx = i * stride
                    end_idx = min(start_idx + window_out.size(1), N)
                    
                    # Accumulate outputs and counts
                    local_out[:, start_idx:end_idx] += window_out[:, :end_idx-start_idx]
                    overlap_count[start_idx:end_idx] += 1
                
                # Average overlapping regions
                overlap_count = torch.clamp(overlap_count, min=1)
                local_out = local_out / overlap_count.unsqueeze(0).unsqueeze(-1)
            else:
                local_out = local_outputs[0] if local_outputs else self.local_proj(x)
                # Pad if window output is shorter than original
                if local_out.size(1) < N:
                    remaining = x[:, local_out.size(1):]
                    remaining_out = self.local_proj(remaining)
                    local_out = torch.cat([local_out, remaining_out], dim=1)
        else:
            # For short sequences, apply attention directly
            local_qkv = self.local_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            local_qkv = local_qkv.permute(2, 0, 3, 1, 4)
            local_q, local_k, local_v = local_qkv.unbind(0)
            
            local_attn = (local_q @ local_k.transpose(-2, -1)) * self.scale
            local_attn = F.softmax(local_attn, dim=-1)
            
            local_out = (local_attn @ local_v).transpose(1, 2).reshape(B, N, C)
            local_out = self.local_proj(local_out)
        
        # Improved global attention: chunked processing to avoid OOM
        if N > 1000:  # Chunked global attention for large point clouds
            chunk_size = 500
            global_outputs = []
            
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                x_chunk = x[:, i:end_i]
                
                global_qkv_chunk = self.global_qkv(x_chunk).reshape(
                    B, end_i - i, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                global_q_chunk, global_k_chunk, global_v_chunk = global_qkv_chunk.unbind(0)
                
                # Interact with global K, V
                global_k_all = self.global_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)[:, :, 1]
                global_v_all = self.global_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)[:, :, 2]
                
                # Linear attention
                global_q_chunk = F.relu(global_q_chunk) + self.eps
                global_k_all = F.relu(global_k_all) + self.eps
                
                k_cumsum = global_k_all.sum(dim=-2, keepdim=True)
                denominator = (global_q_chunk * k_cumsum).sum(dim=-1, keepdim=True)
                denominator = torch.clamp(denominator, min=self.eps)
                D_inv = torch.clamp(1. / denominator, max=1e6)
                
                context = global_k_all.transpose(-2, -1) @ global_v_all
                global_attn_chunk = global_q_chunk @ context
                global_attn_chunk = global_attn_chunk * D_inv
                
                global_outputs.append(global_attn_chunk)
            
            global_attn = torch.cat(global_outputs, dim=2)
        else:
            # Original global attention logic
            global_qkv = self.global_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            global_q, global_k, global_v = global_qkv.unbind(0)
            
            global_q = F.relu(global_q) + self.eps
            global_k = F.relu(global_k) + self.eps
            
            k_cumsum = global_k.sum(dim=-2, keepdim=True)
            denominator = (global_q * k_cumsum).sum(dim=-1, keepdim=True)
            denominator = torch.clamp(denominator, min=self.eps)
            D_inv = torch.clamp(1. / denominator, max=1e6)
            
            context = global_k.transpose(-2, -1) @ global_v
            global_attn = global_q @ context
            global_attn = global_attn * D_inv
        
        global_out = global_attn.transpose(1, 2).reshape(B, N, C)
        global_out = self.global_proj(global_out)
        
        # Fuse local and global features
        fused = torch.cat([local_out, global_out], dim=-1)
        output = self.fusion(fused)
        
        return output

class MinkAttentionBlock(nn.Module):
    """Minkowski Attention Block - sparse tensor compatible"""
    def __init__(self, channels, attention_type='eca', use_stable_attention=True, **kwargs):
        super(MinkAttentionBlock, self).__init__()
        self.attention_type = attention_type
        
        if attention_type == 'eca':
            self.attention = ECAAttention(channels)
        elif attention_type == 'se':
            self.attention = SEAttention(channels, kwargs.get('reduction', 16))
        elif attention_type == 'simam':
            self.attention = SimAM(kwargs.get('e_lambda', 1e-4))
        elif attention_type == 'linear':
            if use_stable_attention:
                self.attention = StableLinearAttention(channels, kwargs.get('num_heads', 8))
            else:
                self.attention = LinearAttention(channels, kwargs.get('num_heads', 8))
        elif attention_type == 'local_global':
            if use_stable_attention:
                self.attention = StableLocalGlobalAttention(channels, kwargs.get('num_heads', 8),
                                                    kwargs.get('window_size', 16))
            else:
                self.attention = LocalGlobalAttention(channels, kwargs.get('num_heads', 8),
                                                    kwargs.get('window_size', 16))
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        """
        x: SparseTensor
        """
        # Extract features and coordinates
        features = x.F  # [N, C]
        coords = x.C.float() if hasattr(x, 'C') else None  # [N, 4] (batch_idx, x, y, z)
        
        if self.attention_type in ['linear', 'local_global']:
            # For transformer-type attention, need sequence format
            # Treat all points as a single sequence
            features = features.unsqueeze(0)  # [1, N, C]
            features = self.norm(features)
            
            # Pass coordinate info for local-global attention
            if self.attention_type == 'local_global' and coords is not None:
                # Extract spatial coordinates (remove batch dim)
                spatial_coords = coords[:, 1:].unsqueeze(0)  # [1, N, 3]
                attended_features = self.attention(features, spatial_coords)
            else:
                attended_features = self.attention(features)
            
            attended_features = attended_features.squeeze(0)  # [N, C]
        else:
            # For channel attention, apply directly
            attended_features = self.attention(features)
        
        # Create new sparse tensor preserving original attributes
        return ME.SparseTensor(attended_features, 
                             coordinate_map_key=x.coordinate_map_key,
                             coordinate_manager=x.coordinate_manager,
                             tensor_stride=x.tensor_stride)

# =============================================================================
# Transformer-enhanced MinkUNet
# =============================================================================

class MinkTransformerBlock(nn.Module):
    """Lightweight Transformer Block"""
    def __init__(self, channels, num_heads=8, mlp_ratio=4., qkv_bias=False, 
                 attention_type='linear', use_stable_attention=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        
        # Select attention type
        if attention_type == 'linear':
            if use_stable_attention:
                self.attn = LinearAttention(channels, num_heads, qkv_bias)
            else:
                self.attn = LinearAttention(channels, num_heads, qkv_bias)
        elif attention_type == 'local_global':
            if use_stable_attention:
                self.attn = LocalGlobalAttention(channels, num_heads, qkv_bias=qkv_bias)
            else:
                self.attn = LocalGlobalAttention(channels, num_heads, qkv_bias=qkv_bias)
        else:
            raise ValueError(f"Unsupported attention type for transformer: {attention_type}")
        
        self.norm2 = nn.LayerNorm(channels)
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, channels)
        )

    def forward(self, x):
        """
        x: [B, N, C] or [N, C]
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # [1, N, C]
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Self-attention
        x = x + self.attn(self.norm1(x))
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        if squeeze_output:
            x = x.squeeze(0)  # [N, C]
            
        return x

class MinkUNetTransformerBase(nn.Module):
    """Transformer-enhanced MinkUNet base class"""
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = None
    PLANES = None
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    def __init__(self, in_channels, out_channels, D=3, 
                 attention_type='eca', transformer_layers=2, num_heads=8,
                 use_transformer=False, use_attention=True, use_stable_attention=True):
        super().__init__()
        self.D = D
        self.attention_type = attention_type
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.use_transformer = use_transformer
        self.use_attention = use_attention
        self.use_stable_attention = use_stable_attention
        
        assert self.BLOCK is not None
        assert self.LAYERS is not None, "LAYERS must be defined in subclass"
        assert self.PLANES is not None, "PLANES must be defined in subclass"
        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        # Encoder
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])
        
        # Add attention
        if self.use_attention:
            self.attn1 = MinkAttentionBlock(self.PLANES[0] * self.BLOCK.expansion, 
                                           self.attention_type, self.use_stable_attention, num_heads=self.num_heads)

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])
        
        if self.use_attention:
            self.attn2 = MinkAttentionBlock(self.PLANES[1] * self.BLOCK.expansion, 
                                           self.attention_type, self.use_stable_attention, num_heads=self.num_heads)

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])
        
        if self.use_attention:
            self.attn3 = MinkAttentionBlock(self.PLANES[2] * self.BLOCK.expansion, 
                                           self.attention_type, self.use_stable_attention, num_heads=self.num_heads)

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])
        
        if self.use_transformer:
            self.transformer_blocks = nn.ModuleList([
                MinkTransformerBlock(self.PLANES[3] * self.BLOCK.expansion, 
                                   num_heads=self.num_heads, 
                                   attention_type='linear' if self.attention_type in ['linear', 'local_global'] else 'linear',
                                   use_stable_attention=self.use_stable_attention)
                for _ in range(self.transformer_layers)
            ])

        # Decoder
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])
        
        if self.use_attention:
            self.attn5 = MinkAttentionBlock(self.PLANES[4] * self.BLOCK.expansion, 
                                           self.attention_type, self.use_stable_attention, num_heads=self.num_heads)

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])
        
        if self.use_attention:
            self.attn6 = MinkAttentionBlock(self.PLANES[5] * self.BLOCK.expansion, 
                                           self.attention_type, self.use_stable_attention, num_heads=self.num_heads)

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])
        
        if self.use_attention:
            self.attn7 = MinkAttentionBlock(self.PLANES[6] * self.BLOCK.expansion, 
                                           self.attention_type, num_heads=self.num_heads)

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.final_sematic = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )
        return nn.Sequential(*layers)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        if self.use_attention:
            out_b1p2 = self.attn1(out_b1p2)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        if self.use_attention:
            out_b2p4 = self.attn2(out_b2p4)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        if self.use_attention:
            out_b3p8 = self.attn3(out_b3p8)

        # Bottleneck
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out) #torch.Size([671, 256])
        
        # Transformer blocks
        if self.use_transformer and hasattr(self, 'transformer_blocks'):
            # Extract features for Transformer processing
            transformer_features = out.F  # [N, C]
            for transformer_block in self.transformer_blocks:
                transformer_features = transformer_block(transformer_features)
            # Create new sparse tensor
            out = ME.SparseTensor(transformer_features, 
                                coordinate_map_key=out.coordinate_map_key,
                                coordinate_manager=out.coordinate_manager,
                                tensor_stride=out.tensor_stride)

        # Decoder path
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out_4 = self.relu(out)

        out = ME.cat(out_4, out_b3p8)
        out = self.block5(out)
        if self.use_attention:
            out = self.attn5(out)

        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)
        if self.use_attention:
            out = self.attn6(out)

        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)
        if self.use_attention:
            out = self.attn7(out)

        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        backbone_feat = self.final_sematic(out)
        return backbone_feat

# =============================================================================
# Transformer UNet variants
# =============================================================================

class MinkUNet34Transformer(MinkUNetTransformerBase):
    """MinkUNet34 + Transformer"""
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

class MinkUNet34TransformerECA(MinkUNet34Transformer):
    """MinkUNet34 + ECA Attention + Transformer"""
    def __init__(self, in_channels, out_channels, D=3, transformer_layers=2, num_heads=8):
        super().__init__(in_channels, out_channels, D, 
                        attention_type='eca', transformer_layers=transformer_layers, 
                        num_heads=num_heads, use_transformer=True, use_attention=True)

class MinkUNet34TransformerSE(MinkUNet34Transformer):
    """MinkUNet34 + SE Attention + Transformer"""
    def __init__(self, in_channels, out_channels, D=3, transformer_layers=2, num_heads=8):
        super().__init__(in_channels, out_channels, D, 
                        attention_type='se', transformer_layers=transformer_layers, 
                        num_heads=num_heads, use_transformer=True, use_attention=True)

class MinkUNet34TransformerSimAM(MinkUNet34Transformer):
    """MinkUNet34 + SimAM Attention + Transformer"""
    def __init__(self, in_channels, out_channels, D=3, transformer_layers=2, num_heads=8):
        super().__init__(in_channels, out_channels, D, 
                        attention_type='simam', transformer_layers=transformer_layers, 
                        num_heads=num_heads, use_transformer=True, use_attention=True)

class MinkUNet34TransformerLinear(MinkUNet34Transformer):
    """MinkUNet34 + Linear Attention + Transformer"""
    def __init__(self, in_channels, out_channels, D=3, transformer_layers=2, num_heads=8):
        super().__init__(in_channels, out_channels, D, 
                        attention_type='linear', transformer_layers=transformer_layers, 
                        num_heads=num_heads, use_transformer=True, use_attention=True)


# Lightweight versions
class MinkUNet18Transformer(MinkUNetTransformerBase):
    """Lightweight MinkUNet18 + Transformer"""
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)

class MinkUNet18TransformerECA(MinkUNet18Transformer):
    """Lightweight MinkUNet18 + ECA Attention + Transformer"""
    def __init__(self, in_channels, out_channels, D=3, transformer_layers=1, num_heads=4):
        super().__init__(in_channels, out_channels, D, 
                        attention_type='eca', transformer_layers=transformer_layers, 
                        num_heads=num_heads, use_transformer=True, use_attention=True)

# =============================================================================
# Factory function
# =============================================================================

def MinkUNet_Transformer(in_channels=3, out_channels=32, D=3, arch='MinkUNet34TransformerECA', 
                        transformer_layers=2, num_heads=8):
    """
    Create Transformer-enhanced MinkUNet
    
    Args:
        arch: architecture selection
            - MinkUNet34TransformerECA: MinkUNet34 + ECA Attention + Transformer
            - MinkUNet34TransformerSE: MinkUNet34 + SE Attention + Transformer  
            - MinkUNet34TransformerSimAM: MinkUNet34 + SimAM Attention + Transformer
            - MinkUNet34TransformerLinear: MinkUNet34 + Linear Attention + Transformer
            - MinkUNet34TransformerLocalGlobal: MinkUNet34 + Local-Global Attention + Transformer
            - MinkUNet18TransformerECA: Lightweight version
        transformer_layers: number of Transformer layers
        num_heads: number of attention heads
    """
    if arch == 'MinkUNet34TransformerECA':
        return MinkUNet34TransformerECA(in_channels, out_channels, D, transformer_layers, num_heads)
    elif arch == 'MinkUNet34TransformerSE':
        return MinkUNet34TransformerSE(in_channels, out_channels, D, transformer_layers, num_heads)
    elif arch == 'MinkUNet34TransformerSimAM':
        return MinkUNet34TransformerSimAM(in_channels, out_channels, D, transformer_layers, num_heads)
    elif arch == 'MinkUNet34TransformerLinear':
        return MinkUNet34TransformerLinear(in_channels, out_channels, D, transformer_layers, num_heads)
    elif arch == 'MinkUNet34TransformerLocalGlobal':
        return MinkUNet34TransformerLocalGlobal(in_channels, out_channels, D, transformer_layers, num_heads)
    elif arch == 'MinkUNet34TransformerLocalGlobalStable':
        return MinkUNet34TransformerLocalGlobal(in_channels, out_channels, D, transformer_layers, num_heads)
    elif arch == 'MinkUNet34TransformerLocalGlobalStableV2':
        return MinkUNet34TransformerLocalGlobalStableV2(in_channels, out_channels, D, transformer_layers, num_heads)
    elif arch == 'MinkUNet18TransformerECA':
        return MinkUNet18TransformerECA(in_channels, out_channels, D, transformer_layers, num_heads)
    else:
        raise ValueError(f'Unsupported architecture: {arch}')

class LinearAttention(nn.Module):
    """Linear Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Linear attention O(n) complexity
        k_cumsum = k.sum(dim=-2, keepdim=True)
        D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)
        
        context = k.transpose(-2, -1) @ v
        attn = q @ context
        attn = attn * D_inv
        
        attn = attn.transpose(1, 2).reshape(B, N, C)
        attn = self.proj(attn)
        return attn


class LocalGlobalAttention(nn.Module):
    """Local-Global Attention - combining local and global information"""
    def __init__(self, dim, num_heads=8, window_size=16, qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Local attention
        self.local_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.local_proj = nn.Linear(dim, dim // 2)
        
        # Global attention (linear attention for reduced complexity)
        self.global_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.global_proj = nn.Linear(dim, dim // 2)
        
        # Fusion layer
        self.fusion = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        
        # Local attention (windowed)
        if N > self.window_size:
            # Split sequence into windows
            num_windows = N // self.window_size
            x_windowed = x[:, :num_windows*self.window_size].view(B, num_windows, self.window_size, C)
            
            local_qkv = self.local_qkv(x_windowed).reshape(B, num_windows, self.window_size, 3, self.num_heads, C // self.num_heads)
            local_qkv = local_qkv.permute(3, 0, 1, 4, 2, 5)
            local_q, local_k, local_v = local_qkv.unbind(0)
            
            local_attn = (local_q @ local_k.transpose(-2, -1)) * self.scale
            local_attn = F.softmax(local_attn, dim=-1)
            
            local_out = (local_attn @ local_v).transpose(2, 3).reshape(B, num_windows, self.window_size, C)
            local_out = local_out.reshape(B, num_windows*self.window_size, C)
            local_out = self.local_proj(local_out)
            
            # Process remaining points
            if N > num_windows * self.window_size:
                remaining = x[:, num_windows*self.window_size:]
                remaining_out = self.local_proj(remaining)
                local_out = torch.cat([local_out, remaining_out], dim=1)
        else:
            local_out = self.local_proj(x)
        
        # Global attention (linear attention)
        global_qkv = self.global_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        global_q, global_k, global_v = global_qkv.unbind(0)
        
        # Use linear attention
        global_q = F.elu(global_q) + 1
        global_k = F.elu(global_k) + 1
        
        k_cumsum = global_k.sum(dim=-2, keepdim=True)
        D_inv = 1. / (global_q * k_cumsum).sum(dim=-1, keepdim=True)
        
        context = global_k.transpose(-2, -1) @ global_v
        global_attn = global_q @ context
        global_attn = global_attn * D_inv
        
        global_out = global_attn.transpose(1, 2).reshape(B, N, C)
        global_out = self.global_proj(global_out)
        
        # Fuse local and global features
        fused = torch.cat([local_out, global_out], dim=-1)
        output = self.fusion(fused)
        
        return output 

# Local-Global attention
class MinkUNet34TransformerLocalGlobal(MinkUNet34Transformer):
    """MinkUNet34 + Local-Global Attention + Transformer"""
    def __init__(self, in_channels, out_channels, D=3, transformer_layers=2, num_heads=8):
        super().__init__(in_channels, out_channels, D, 
                        attention_type='local_global', transformer_layers=transformer_layers, 
                        num_heads=num_heads, use_transformer=True, use_attention=False, 
                        use_stable_attention=True)

