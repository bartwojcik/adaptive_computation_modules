from functools import partial

import torch
import torchvision
from torchvision.models import ViT_B_16_Weights


def get_vit_b_16():
    model = torchvision.models.vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1, progress=False)

    def forward_generator(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # BEGIN ENCODER
        # equivalent to: x = self.encoder(x)
        x = x + self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        # go through encoder blocks
        for block in self.encoder.layers:
            x = block(x)
            x = yield x, None
        x = self.encoder.ln(x)
        # END OF ENCODER
        # classifier token
        x = x[:, 0]
        x = self.heads(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = 224
    model.input_channels = 3
    model.number_of_classes = 1000
    return model
