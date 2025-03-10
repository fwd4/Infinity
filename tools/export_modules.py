from infinity.models.basic import *
from infinity.models.infinity import TextAttentivePool
import numpy as np

seqlen = 35

class TextPre(torch.nn.Module):
    def __init__(self):
        self.Ct5 = 2048 # channel of T5 model
        self.D = 2048
        self.norm_eps = 1e-6

        super().__init__()
        self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=self.norm_eps)
        self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
    
    def forward(self, kv_compact): 
        kv_compact = self.text_norm(kv_compact)
        cu_seqlens_k = np.array([0, seqlen, seqlen*2])
        max_seqlen_k = seqlen
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))    # cond_BD should be float32
        return kv_compact
       

if __name__ == '__main__':
    text_pre = TextPre()
    torch_input = torch.randn(2*seqlen, 2048)
    onnx_program = torch.onnx.dynamo_export(text_pre, torch_input)
    onnx_program.save('text_preprocess.onnx')

