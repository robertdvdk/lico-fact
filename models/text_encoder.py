import os
import torch
import torch.nn as nn
from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu():
    backbone_name = "ViT-B/32"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root=os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        clip_model = clip_model.type(torch.FloatTensor)
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # eot_token is the fullstop (end of text)
        # they want to condense the transformer features self.transformer(x) to a single vector and
        # the eot token acts as a summarization vector of the whole input prompt's features
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 12
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.N = 1

        # random initialization
        print("Initializing class-specific contexts")
        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)  # define the prompt to be trained
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # Tokenized prompts are whole sentences like "XXXXXXXX <label>."
        # Here they are turned to clip vocabulary tokens
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (10, 77)
        # This adds an extra dimension of N, repeting tokenized_prompts along it
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)

        with torch.no_grad():
            # Embedded vectors of tokenized_prompts
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        print('tokenized prompts:', embedding.shape, 'ctx: ', self.ctx.shape)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # token prefix is embedded("XXXXXXXX <label>.")[0] , which is just embedded("X")
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # token_suffix is embedded("<label>.")
        # EOS token is the full stop
        # we need it for rendering a matrix of vectors to a single feature vector later in TextEncoder
        # it is like a cls token in ViT (my interpretation)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def _ctx_shuffle(self, ctx):
        # shuffle the ctx along 2nd dimension
        rand_idx = torch.randperm(ctx.shape[1])
        shuffled_ctx = ctx[:, rand_idx, :]
        return shuffled_ctx

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0)

        ctx = ctx.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx = self._ctx_shuffle(prefix, suffix, ctx)

        # This if is about different ordering of the prefix, ctx and suffix
        half_n_ctx = self.n_ctx // 2
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = prefix[i: i + 1, :, :]
            class_i = suffix[i: i + 1, :name_len, :]
            suffix_i = suffix[i: i + 1, name_len:, :]
            ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
            prompt = torch.cat(
                [
                    prefix_i,  # (1, 1, dim)
                    ctx_i_half1,  # (1, n_ctx//2, dim)
                    class_i,  # (1, name_len, dim)
                    ctx_i_half2,  # (1, n_ctx//2, dim)
                    suffix_i,  # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)
        return prompts

if __name__ == '__main__':
    pass