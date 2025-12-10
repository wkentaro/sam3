import types
import onnx
import onnxruntime
import torch
import numpy as np
from PIL import Image
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from torchvision.transforms import v2
from osam._models.yoloworld.clip import tokenize
import imgviz

from infer_torch import get_replace_freqs_cis


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._model: Sam3Image = build_sam3_image_model()
        get_replace_freqs_cis(self._model)

        # self._processor: Sam3Processor = Sam3Processor(self._model)
        self.transform = v2.Compose(
            [
                # NOTE: Resize in .transform has difference between pytorch and onnx
                # v2.ToDtype(torch.uint8, scale=True),
                # v2.Resize(size=(resolution := 1008, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def forward(self, image: torch.Tensor) -> list[str, torch.Tensor]:
        # image = self._processor.transform(image).unsqueeze(0)
        image = self.transform(image).unsqueeze(0)

        backbone_out = self._model.backbone._forward_image_no_act_ckpt(image)
        del backbone_out["vision_features"]
        del backbone_out["sam2_backbone_out"]

        assert len(backbone_out["vision_pos_enc"]) == 3
        assert len(backbone_out["backbone_fpn"]) == 3
        return [*backbone_out["vision_pos_enc"], *backbone_out["backbone_fpn"]]


def export_image_backbone(image: Image):
    image = image.resize((1008, 1008), resample=Image.BILINEAR)
    image = v2.functional.to_image(image).to("cuda")

    if 0:
        encoder = ImageEncoder()
        with torch.no_grad():
            output_ = encoder(image)

        torch.onnx.export(
            encoder,
            args=(image,),
            f="sam3_image_encoder.onnx",
            input_names=["image"],
            output_names=[
                "vision_pos_enc.0",
                "vision_pos_enc.1",
                "vision_pos_enc.2",
                "backbone_fpn.0",
                "backbone_fpn.1",
                "backbone_fpn.2",
            ],
            # dynamic_axes={
            #     "image": {1: "height", 2: "width"},
            # },
            opset_version=21,
        )
        print("exported onnx model")

        onnx_model = onnx.load("sam3_image_encoder.onnx")
        onnx.checker.check_model(onnx_model)
        print("exported model is valid")

    session = onnxruntime.InferenceSession("sam3_image_encoder.onnx")
    output = session.run(None, {"image": image.cpu().numpy()})
    print("image onnx runtime inference done")

    return output


class TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._model: Sam3Image = build_sam3_image_model()

    def VETextEncoder_forward(self, tokenized: torch.Tensor) -> torch.Tensor:
        # VETextEncoder.forward
        text_attention_mask = (tokenized != 0).bool()

        inputs_embeds = self._model.backbone.language_backbone.encoder.token_embedding(
            tokenized
        )
        _, text_memory = self._model.backbone.language_backbone.encoder(tokenized)

        assert text_memory.shape[1] == inputs_embeds.shape[1]
        # Invert attention mask because its the opposite in pytorch transformer
        text_attention_mask = text_attention_mask.ne(1)
        # Transpose memory because pytorch's attention expects sequence first
        text_memory = text_memory.transpose(0, 1)
        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = self._model.backbone.language_backbone.resizer(
            text_memory
        )
        return text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1)

    def forward(self, tokenized: torch.Tensor) -> torch.Tensor:
        # SAM3VLBackbone.forward
        return self.VETextEncoder_forward(tokenized)


def export_text_backbone():
    tokenized = tokenize(["person"], context_length=32)
    tokenized = torch.from_numpy(tokenized).to("cuda")

    if 0:
        encoder = TextEncoder()
        with torch.no_grad():
            output_ = encoder(tokenized)

        torch.onnx.export(
            encoder,
            args=(tokenized,),
            f="sam3_text_encoder.onnx",
            input_names=["tokenized"],
            output_names=["text_attention_mask", "text_memory", "text_embeds"],
            # dynamic_axes={
            #     "captions": {0: "batch_size", 1: "seq_len"},
            #     "text_memory_resized": {0: "batch_size", 1: "seq_len"},
            # },
            opset_version=21,
        )
        print("exported onnx model")

        onnx_model = onnx.load("sam3_text_encoder.onnx")
        onnx.checker.check_model(onnx_model)
        print("exported model is valid")

    session = onnxruntime.InferenceSession("sam3_text_encoder.onnx")
    output = session.run(None, {"tokenized": tokenized.cpu().numpy()})
    print("text onnx runtime inference done")

    return output


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._model: Sam3Image = build_sam3_image_model()
        self._processor: Sam3Processor = Sam3Processor(self._model)

    def forward(
        self,
        original_height: torch.Tensor,
        original_width: torch.Tensor,
        vision_pos_enc_0: torch.Tensor,
        vision_pos_enc_1: torch.Tensor,
        vision_pos_enc_2: torch.Tensor,
        backbone_fpn_0: torch.Tensor,
        backbone_fpn_1: torch.Tensor,
        backbone_fpn_2: torch.Tensor,
        language_mask: torch.Tensor,
        language_features: torch.Tensor,
        language_embeds: torch.Tensor,
    ) -> torch.Tensor:
        device = vision_pos_enc_0.device
        point_embeddings = torch.empty((0, 1, 2), device=device)
        point_mask = torch.empty((1, 0), device=device, dtype=torch.bool)
        point_labels = torch.empty((0, 1), device=device, dtype=torch.int64)
        box_embeddings = torch.empty((0, 1, 4), device=device)
        box_mask = torch.empty((1, 0), device=device, dtype=torch.bool)
        box_labels = torch.empty((0, 1), device=device, dtype=torch.int64)
        mask_mask = torch.empty((1, 0), device=device, dtype=torch.bool)
        mask_labels = torch.empty((0, 1), device=device, dtype=torch.int64)

        state = {
            "original_height": original_height,
            "original_width": original_width,
            "backbone_out": {
                "vision_pos_enc": [
                    vision_pos_enc_0,
                    vision_pos_enc_1,
                    vision_pos_enc_2,
                ],
                "backbone_fpn": [
                    backbone_fpn_0,
                    backbone_fpn_1,
                    backbone_fpn_2,
                ],
                "language_mask": language_mask,
                "language_features": language_features,
                "language_embeds": language_embeds,
            },
            "geometric_prompt": self._processor.model._get_dummy_prompt(),
        }
        # prompt, prompt_mask, _ = self._processor.model._encode_prompt(
        #     backbone_out=state["backbone_out"],
        #     find_input=self._processor.find_stage,
        #     geometric_prompt=state["geometric_prompt"],
        # )
        # return prompt, prompt_mask
        result = self._processor._forward_grounding(state)
        return result["boxes"], result["scores"], result["masks"]


def export_decoder(state: dict):
    if 0:
        decoder = Decoder()
        with torch.no_grad():
            output_ = decoder(
                original_height=torch.tensor(state["original_height"])
                .unsqueeze(0)
                .to("cuda"),
                original_width=torch.tensor(state["original_width"])
                .unsqueeze(0)
                .to("cuda"),
                vision_pos_enc_0=state["backbone_out"]["vision_pos_enc"][0],
                vision_pos_enc_1=state["backbone_out"]["vision_pos_enc"][1],
                vision_pos_enc_2=state["backbone_out"]["vision_pos_enc"][2],
                backbone_fpn_0=state["backbone_out"]["backbone_fpn"][0],
                backbone_fpn_1=state["backbone_out"]["backbone_fpn"][1],
                backbone_fpn_2=state["backbone_out"]["backbone_fpn"][2],
                language_mask=state["backbone_out"]["language_mask"],
                language_features=state["backbone_out"]["language_features"],
                language_embeds=state["backbone_out"]["language_embeds"],
            )

        torch.onnx.export(
            decoder,
            args=(
                torch.tensor(state["original_height"]).unsqueeze(0).to("cuda"),
                torch.tensor(state["original_width"]).unsqueeze(0).to("cuda"),
                state["backbone_out"]["vision_pos_enc"][0],
                state["backbone_out"]["vision_pos_enc"][1],
                state["backbone_out"]["vision_pos_enc"][2],
                state["backbone_out"]["backbone_fpn"][0],
                state["backbone_out"]["backbone_fpn"][1],
                state["backbone_out"]["backbone_fpn"][2],
                state["backbone_out"]["language_mask"],
                state["backbone_out"]["language_features"],
                state["backbone_out"]["language_embeds"],
            ),
            f="sam3_decoder.onnx",
            input_names=[
                "original_height",
                "original_width",
                "vision_pos_enc.0",
                "vision_pos_enc.1",
                "vision_pos_enc.2",
                "backbone_fpn.0",
                "backbone_fpn.1",
                "backbone_fpn.2",
                "language_mask",
                "language_features",
                "language_embeds",
            ],
            output_names=["boxes", "scores", "masks"],
            opset_version=21,
            verify=True,
        )
        print("exported onnx model")

        # onnx_model = onnx.load("sam3_decoder.onnx")
        # onnx.checker.check_model(onnx_model)
        # print("exported model is valid")

    session = onnxruntime.InferenceSession("sam3_decoder.onnx")
    output = session.run(
        None,
        {
            "original_height": np.array([state["original_height"]]),
            "original_width": np.array([state["original_width"]]),
            "vision_pos_enc.0": state["backbone_out"]["vision_pos_enc"][0]
            .cpu()
            .numpy(),
            "vision_pos_enc.1": state["backbone_out"]["vision_pos_enc"][1]
            .cpu()
            .numpy(),
            "vision_pos_enc.2": state["backbone_out"]["vision_pos_enc"][2]
            .cpu()
            .numpy(),
            "backbone_fpn.0": state["backbone_out"]["backbone_fpn"][0].cpu().numpy(),
            "backbone_fpn.1": state["backbone_out"]["backbone_fpn"][1].cpu().numpy(),
            "backbone_fpn.2": state["backbone_out"]["backbone_fpn"][2].cpu().numpy(),
            "language_mask": state["backbone_out"]["language_mask"].cpu().numpy(),
            "language_features": state["backbone_out"]["language_features"]
            .cpu()
            .numpy(),
            "language_embeds": state["backbone_out"]["language_embeds"]
            .cpu()
            .numpy(),
        },
    )
    print("decoder onnx runtime inference done")

    return output


def main():
    # image {{
    # image = Image.open("bus.jpg")
    image = Image.open("2011_000006.jpg")

    image_backbone_out = export_image_backbone(image)

    # processor: Sam3Processor = Sam3Processor(model=build_sam3_image_model())
    # state = processor.set_image(image)
    state = {
        "original_height": image.height,
        "original_width": image.width,
        "backbone_out": {
            "vision_pos_enc": [
                torch.from_numpy(image_backbone_out[0]).to("cuda"),
                torch.from_numpy(image_backbone_out[1]).to("cuda"),
                torch.from_numpy(image_backbone_out[2]).to("cuda"),
            ],
            "backbone_fpn": [
                torch.from_numpy(image_backbone_out[3]).to("cuda"),
                torch.from_numpy(image_backbone_out[4]).to("cuda"),
                torch.from_numpy(image_backbone_out[5]).to("cuda"),
            ],
        },
    }
    # }}

    # text {{
    text_backbone_out = export_text_backbone()

    # result = processor.set_text_prompt(prompt="person", state=state)
    state["backbone_out"] |= {
        "language_mask": torch.from_numpy(text_backbone_out[0]).to("cuda"),
        "language_features": torch.from_numpy(text_backbone_out[1]).to("cuda"),
        "language_embeds": torch.from_numpy(text_backbone_out[2]).to("cuda"),
    }
    # }}

    # grounding {{
    boxes, scores, masks = export_decoder(state=state)

    # result = processor._forward_grounding(state)
    # }}

    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks[:, 0, :, :],
        bboxes=boxes[:, [1, 0, 3, 2]],
        labels=np.arange(len(boxes)),
        captions=[f"{s:.2f}" for s in scores],
    )
    imgviz.io.pil_imshow(viz)


if __name__ == "__main__":
    main()
