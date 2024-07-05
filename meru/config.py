from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder
from meru.models import MERU, CLIPBaseline


def prepare_model(train_config):
    textual = TransformerTextEncoder(arch="L12_W512", vocab_size=49408, context_length=77)
    if train_config == 'clip_vit_l':
        visual = build_timm_vit(arch="vit_large_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = CLIPBaseline(visual= visual, textual = textual, embed_dim = 512)
    elif train_config == 'clip_vit_b':
        visual = build_timm_vit(arch="vit_base_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = CLIPBaseline(visual= visual, textual = textual, embed_dim = 512)
    elif train_config == 'clip_vit_s':
        visual = build_timm_vit(arch="vit_small_mocov3_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = CLIPBaseline(visual= visual, textual = textual, embed_dim = 512)
    elif train_config == 'meru_vit_l':
        visual = build_timm_vit(arch="vit_large_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = MERU(visual=visual, textual=textual, embed_dim=512, curv_init=1.0, learn_curv=True, entail_weight=0.2,)
    elif train_config == 'meru_vit_b':
        visual = build_timm_vit(arch="vit_base_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = MERU(visual=visual, textual=textual, embed_dim=512, curv_init=1.0, learn_curv=True, entail_weight=0.2,)
    elif train_config == 'meru_vit_s':
        visual = build_timm_vit(arch="vit_small_mocov3_patch16_224", global_pool="token", use_sincos2d_pos=True)
        model = MERU(visual=visual, textual=textual, embed_dim=512, curv_init=1.0, learn_curv=True, entail_weight=0.2,)
    else:
        raise NotImplementedError
    return model
