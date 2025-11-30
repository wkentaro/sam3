import imgviz
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def get_replace_freqs_cis(module):
    if hasattr(module, "freqs_cis"):
        freqs_cos = module.freqs_cis.real.float()
        freqs_sin = module.freqs_cis.imag.float()
        # Replace the buffer
        module.register_buffer("freqs_cos", freqs_cos)
        module.register_buffer("freqs_sin", freqs_sin)
        del module.freqs_cis  # Remove complex version
    for child in module.children():
        get_replace_freqs_cis(child)


def main():
    image = Image.open("bus.jpg")

    model: Sam3Image = build_sam3_image_model()
    with torch.no_grad():
        get_replace_freqs_cis(model)

    processor: Sam3Processor = Sam3Processor(model)
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt="person", state=state)

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks.cpu().numpy()[:, 0, :, :],
        bboxes=boxes.cpu().numpy()[:, [1, 0, 3, 2]],
        labels=np.arange(len(masks)),
        captions=[f"{s:.2f}" for s in scores],
    )
    imgviz.io.pil_imshow(viz)


if __name__ == "__main__":
    main()
