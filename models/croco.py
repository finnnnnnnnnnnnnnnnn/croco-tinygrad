import tinygrad
import tinygrad.nn as nn
from models.blocks import PatchEmbed, Block, DecoderBlock
from models.masking import RandomMask
from models.pos_embed import get_2d_sincos_pos_embed, RoPE2D
import numpy as np
from models.util import normal_init, xavier_uniform
from functools import partial

class CroCoNet():
    def __init__(self,
                 img_size=224,           # input image size
                 patch_size=16,          # patch_size 
                 mask_ratio=0.9,         # ratios of masked tokens 
                 enc_embed_dim=768,      # encoder feature dimension
                 enc_depth=12,           # encoder depth 
                 enc_num_heads=12,       # encoder number of heads in the transformer block 
                 dec_embed_dim=512,      # decoder feature dimension 
                 dec_depth=8,            # decoder depth 
                 dec_num_heads=16,       # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='cosine',     # positional embedding (either cosine or RoPE100)
                ):
                
        super(CroCoNet, self).__init__()
                
        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)

        # mask generations
        self._set_mask_generator(self.patch_embed.num_patches, mask_ratio)


        self.pos_embed = pos_embed
        if pos_embed=='cosine':
            # positional embedding of the encoder 
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
            self.enc_pos_embed = tinygrad.Tensor(enc_pos_embed.astype(np.float32)) # <-change to convert numpy array directly to Tensor
            # positional embedding of the decoder  
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
            self.dec_pos_embed = tinygrad.Tensor(dec_pos_embed.astype(np.float32)) # <-change to convert numpy array directly to Tensor
            # pos embedding in each block
            self.rope = None # nothing for cosine 
        elif pos_embed.startswith('RoPE'): # eg RoPE100
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)

        # transformer for the encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = [
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)]
        self.enc_norm = norm_layer(enc_embed_dim)
        
        # masked tokens 
        self._set_mask_token(dec_embed_dim)

        # decoder 
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)
        
        # # prediction head 
        self._set_prediction_head(dec_embed_dim, patch_size)
        
        # # initializer weights
        # self.initialize_weights()

    #DONE
    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    #DONE
    def _set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)
    
    #DONE
    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = tinygrad.Tensor.zeros(1, 1, dec_embed_dim) # <- got rid of parameter thing
        
    #DONE
    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder 
        self.dec_blocks = [
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)]
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)
        
    #DONE
    def _set_prediction_head(self, dec_embed_dim, patch_size):
         self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)
        
    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # mask tokens
        if self.mask_token is not None: normal_init(self.mask_token, std=.02)
        # linears and layer norms
        # self.apply(self._init_weights) I guess this dones't make sense to do???

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         # we use xavier_uniform following official JAX ViT:
    #         xavier_uniform(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
            
    def _encode_image(self, image, do_mask=False, return_all_blocks=False):
        """
        image has B x 3 x img_size x img_size 
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # embed the image into patches  (x has size B x Npatches x C) 
        # and get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)              
        # add positional embedding without cls token  
        if self.enc_pos_embed is not None: 
            x = x + self.enc_pos_embed[None,...]
        # apply masking 
        #CLAUD CODE
        # B, N, C = x.size()
        # if do_mask:
        #         masks = self.mask_generator(x)  # This returns boolean tensor
        #         num_unmasked = (masks == 0).sum().numpy()
        #         # Convert boolean mask to indices
        #         indices = tinygrad.helpers.argsort(masks.reshape(-1))[:int(num_unmasked)]
        #         # Reshape x to 2D (flattening batch dimension)
        #         x_flat = x.reshape(-1, C)
        #         pos_flat = pos.reshape(-1, 2)
        #         # Index using integer indices
        #         x = x_flat[indices].reshape(B, -1, C)
        #         posvis = pos_flat[indices].reshape(B, -1, 2)
        # else:
        #     masks = tinygrad.Tensor.zeros((B,N), dtype=tinygrad.dtypes.bool)
        #     posvis = pos
        #CLAUD CODE
        B,N,C = x.size()
        if do_mask:
            masks = self.mask_generator(x)
            x = tinygrad.Tensor(x.numpy()[masks.logical_not().numpy()]).view(B, -1, C)
            posvis = tinygrad.Tensor(pos.numpy()[masks.logical_not().numpy()]).view(B, -1, 2)
        else:
            B,N,C = x.size()
            masks = tinygrad.Tensor.zeros((B,N), dtype=tinygrad.dtypes.bool)
            posvis = pos
        # now apply the transformer encoder and normalization        
        if return_all_blocks:
            out = []
            for blk in self.enc_blocks:
                x = blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos, masks
        else:
            for blk in self.enc_blocks:
                x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks
 
    def _decoder(self, feat1, pos1, masks1, feat2, pos2, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
                           
        masks1 can be None => assume image1 fully visible 
        """
        # encoder to decoder layer 
        visf1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        # append masked tokens to the sequence
        B,Nenc,C = visf1.size()
        if masks1 is None: # downstreams
            f1_ = visf1
        else: # pretraining 
            Ntotal = masks1.size(1)
            f1_ = self.mask_token.repeat(B, Ntotal, 1)#.to(dtype=visf1.dtype) -> make need to convert this to nn.Linear

            trues = 0
            for idx, x in enumerate(masks1.logical_not()[0]): 
                if bool(x.numpy()):
                    f1_ = f1_.contiguous()
                    f1_[0][idx] = visf1.view(B * Nenc, C)[trues].contiguous()
                    trues += 1

        # add positional embedding
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed
        # apply Transformer blocks
        out = f1_
        out2 = f2 
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out, out2 = blk(_out, out2, pos1, pos2)
                out.append(_out)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2 = blk(out, out2, pos1, pos2)
            out = self.dec_norm(out)
        return out

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = tinygrad.Tensor.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = tinygrad.Tensor.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs

    def __call__(self, img1, img2):
        """
        img1: tensor of size B x 3 x img_size x img_size
        img2: tensor of size B x 3 x img_size x img_size
        
        out will be    B x N x (3*patch_size*patch_size)
        masks are also returned as B x N just in case 
        """
        # encoder of the masked first image 
        feat1, pos1, mask1 = self._encode_image(img1, do_mask=True)
        # encoder of the second image 
        feat2, pos2, _ = self._encode_image(img2, do_mask=False)
        # decoder 
        decfeat = self._decoder(feat1, pos1, mask1, feat2, pos2)
        # prediction head 
        out = self.prediction_head(decfeat)
        # get target
        target = self.patchify(img1)
        return out, mask1, target 