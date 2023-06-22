# Code based on
# https://github.com/luca-medeiros/lang-segment-anything/blob/main/lang_sam/lang_sam.py
# We are using the new GroundingDINO API to speed up inference
# Similar architecture to https://github.com/IDEA-Research/Grounded-Segment-Anything

import os
import torch
import numpy as np
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.data import MetadataCatalog
from detectron2.config import configurable

from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from huggingface_hub import hf_hub_download
from groundingdino.util.inference import Model
from PIL import Image

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))


def build_sam(sam_type, device):
    checkpoint_url = SAM_MODELS[sam_type]
    try:
        sam = sam_model_registry[sam_type]()
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
        sam.load_state_dict(state_dict, strict=True)
    except:
        raise ValueError(f"Problem loading SAM please make sure you have the right model type: {sam_type} \
            and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
            re-downloading it.")
    sam.to(device=device)
    return SamPredictor(sam)


@META_ARCH_REGISTRY.register()
class GroundedSAM(torch.nn.Module):

    @configurable
    def __init__(self,
                 *,
                 model_name: str,
                 class_names: list,
                 background_class: int = 0,
                 ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = build_sam(model_name, self.device)

        # init grounding DINO
        repo_id = "ShilongLiu/GroundingDINO"
        filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        cache_model = hf_hub_download(repo_id=repo_id, filename=filename)
        self.groundingdino = Model(model_config_path=cache_config_file,
                                   model_checkpoint_path=cache_model,
                                   device=self.device)

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.background_class = background_class
        if self.background_class > self.num_classes:
            # background_class == ignore_label
            self.background_class = self.num_classes


    @classmethod
    def from_config(cls, cfg):
        meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        # Find background class
        if hasattr(meta, 'background_class'):
            background_class = meta.background_class
        elif 'background' in meta.stuff_classes:
            background_class = meta.stuff_classes.index('background')
        elif 'Background' in meta.stuff_classes:
            background_class = meta.stuff_classes.index('Background')
        elif 'others' in meta.stuff_classes:
            background_class = meta.stuff_classes.index('others')
        elif 'Others' in meta.stuff_classes:
            background_class = meta.stuff_classes.index('Others')
        else:
            background_class = 0

        return {
            "model_name": cfg.MODEL.WEIGHTS,
            "class_names": MetadataCatalog.get(cfg.DATASETS.TEST[0]).stuff_classes,
            "background_class": background_class,
        }

    def predict_sam(self, image_pil, boxes):
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
            return_logits=True,
        )
        return masks.cpu()

    def predict_gdino(self, image_pil, text_prompts, box_threshold=0.3, text_threshold=0.25):
        if len(text_prompts) > 50:
            # split up text prompts into batches of 50
            # Grounding DINO has an input limit of 512 tokens. This is a heuristic to avoid hitting that limit
            text_prompts = [text_prompts[i:i + 50] for i in range(0, len(text_prompts), 50)]
            boxes, logits, class_ids = torch.tensor([]), torch.tensor([]), torch.tensor([])
            for i, text_prompts_batch in enumerate(text_prompts):
                b, l, c = self.predict_gdino(image_pil, text_prompts_batch, box_threshold, text_threshold)
                boxes = torch.cat((boxes, b))
                logits = torch.cat((logits, l))
                class_ids = torch.cat((class_ids, c + i * 50))
            return boxes, logits, class_ids

        detections = self.groundingdino.predict_with_classes(np.array(image_pil), text_prompts, box_threshold,
                                                             text_threshold)
        boxes = torch.tensor(detections.xyxy)
        logits = torch.tensor(detections.confidence)
        class_ids = detections.class_id
        class_ids = torch.tensor(class_ids.astype(np.float32))
        return boxes, logits, class_ids

    def predict(self, image_pil, text_prompts):
        boxes, logits, class_ids = self.predict_gdino(image_pil, text_prompts)
        # Grounding DINO sometimes produces None class_ids.
        # These are converted to nan (<= 50 classes) or negative values (> 50 classes) by predict_gdino.
        # We convert them to the background class here and convert the tensor to int
        class_ids.nan_to_num(self.background_class)
        class_ids = class_ids.int()
        class_ids[class_ids < 0] = self.background_class

        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, class_ids, logits

    def forward(self, batched_inputs):
        # get images from batch
        images = [x["image"] for x in batched_inputs]
        # convert images to PIL (Input format of Grounded DINO and SAM)
        images = [Image.fromarray(x.numpy().transpose(1, 2, 0).astype(np.uint8)) for x in images]

        results = []
        for image in images:
            # inference
            masks, boxes, class_id, logits = self.predict(image, self.class_names)
            results.append({
                'masks': masks,
                'boxes': boxes,
                'classes': class_id,
                'logits': logits,
            })

        processed_results = []
        for output, input in zip(results, batched_inputs):
            # multiply SAM mask and with Grounded DINO logits to account for box confidence
            masks = output['masks'] * output['logits'][:, None, None]

            # init prediction mask
            mask_size = input["image"].shape[-2:]
            r = torch.zeros((self.num_classes + 1, *mask_size))

            # convert instance masks to semantic results
            class_ids = output['classes']
            for class_id in class_ids.unique():
                r[class_id] = masks[class_ids == class_id].max(dim=0)[0]

            # set pixels with no prediction (all scores 0. or below) to background class
            r[self.background_class, r.max(dim=0).values <= 0] = 1.

            # resize prediction mask
            height = input.get("height", mask_size[0])
            width = input.get("width", mask_size[1])
            r = sem_seg_postprocess(r, mask_size, height, width)

            # add to processed results
            processed_results.append({
                "sem_seg": r[:-1],
            })

        return processed_results
