import os
import clip
from PIL import Image
from torchvision import transforms
from .constants import CACHE_DIR
from meru.config import prepare_model
from meru.utils.checkpointing import CheckpointManager
import torchvision.transforms as T
from meru import lorentz as L
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer
from torch.nn import functional as F

def get_model(model_name, device, root, root_dir=CACHE_DIR):
    """
    Helper function that returns a model and a potential image preprocessing function.
    """
    if "openai-clip" in model_name:
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, image_preprocess = clip.load(variant, device=device, download_root=root_dir)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif model_name == "clip_vit_l":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(root + '/pretrained_models/meru/clip_vit_l.pth')
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "meru_vit_l":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(root + '/pretrained_models/meru/meru_vit_l.pth')
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "clip_vit_b":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(root + '/pretrained_models/meru/clip_vit_b.pth')
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "meru_vit_b":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(root + '/pretrained_models/meru/meru_vit_b.pth')
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "clip_vit_s":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(root + '/pretrained_models/meru/clip_vit_s.pth')
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "meru_vit_s":
        from .clip_models import CLIPWrapper_hyp
        preprocess = T.Compose([T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(size=(224, 224)), T.ToTensor(), ])
        model = prepare_model(model_name)
        model.cuda()
        CheckpointManager(model=model).load(root + '/pretrained_models/meru/meru_vit_s.pth')
        model.eval()
        clip_model = CLIPWrapper_hyp(model, device)
        return clip_model, preprocess

    elif model_name == "blip-flickr-base":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-flickr-base")
        image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "blip-coco-base":
        from .blip_models import BLIPModelWrapper
        blip_model = BLIPModelWrapper(root_dir=root_dir, device=device, variant="blip-coco-base")
        image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384),interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])  
        return blip_model, image_preprocess
    
    elif model_name == "xvlm-flickr":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-flickr")
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "xvlm-coco":
        from .xvlm_models import XVLMWrapper
        xvlm_model = XVLMWrapper(root_dir=root_dir, device=device, variant="xvlm-coco")
        image_preprocess = transforms.Compose([
                            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
        return xvlm_model, image_preprocess
    
    elif model_name == "flava":
        from .flava import FlavaWrapper
        flava_model = FlavaWrapper(root_dir=root_dir, device=device)
        image_preprocess = None
        return flava_model, image_preprocess

    elif model_name == "NegCLIP":
        import open_clip
        from .clip_models import CLIPWrapper
        
        path = os.path.join(root_dir, "negclip.pth")
        if not os.path.exists(path):
            print("Downloading the NegCLIP model...")
            import gdown
            gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
        model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path, device=device)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess

    elif model_name == "coca":
        import open_clip
        from .clip_models import CLIPWrapper
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name="coca_ViT-B-32", pretrained="laion2B-s13B-b90k", device=device)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    
        
    elif "laion-clip" in model_name:
        import open_clip
        from .clip_models import CLIPWrapper
        variant = model_name.split(":")[1]
        model, _, image_preprocess = open_clip.create_model_and_transforms(model_name=variant, pretrained="laion2b_s34b_b79k", device=device)
        model = model.eval()
        clip_model = CLIPWrapper(model, device) 
        return clip_model, image_preprocess
    
        
    else:
        raise ValueError(f"Unknown model {model_name}")
