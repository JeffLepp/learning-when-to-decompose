"""CPU-only regression tests; optional torch must not break the main env."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v2.probe.extract import ROOT, HYBRID, THINKING, format_prompt, guard_items

try:
    import torch
    import numpy as np
    from transformers import Qwen2Config, Qwen2ForCausalLM
    TORCH = True
except ImportError:
    TORCH = False


class Tokenizer:
    pad_token_id = 0
    padding_side = 'right'

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {'input_ids':[1 + ord(c)%29 for c in text]}


class ContractTests(unittest.TestCase):
    def test_format(self):
        matching = json.loads((ROOT/'v2/probe/model_matching.json').read_text())
        recorded = matching['prompt_format_check']['verified_exact_format']
        self.assertEqual(HYBRID,recorded.replace('{RULES}','{system}').replace('{prompt}','{user}'))
        for model in ['qwen3:0.6b','qwen3:1.7b','qwen3:8b']:
            self.assertEqual(format_prompt(model,'s','u'), HYBRID.format(system='s',user='u'))
        template = next(m['template'] for m in matching['models'] if m['ollama_model']=='qwen3:4b')
        self.assertIn('{{ .System }}\n\n{{ end }}',template)
        self.assertNotIn('/no_think',template)
        self.assertEqual(format_prompt('qwen3:4b','s','u'),THINKING.format(system='s',user='u'))

    def test_ood(self):
        for split in ['ood_length','ood_depth']:
            with patch('v2.probe.extract.subprocess.run') as run:
                run.return_value.stdout = ''
                for flag in [False,True]:
                    with self.assertRaises(ValueError):
                        guard_items([{'split':split}],flag)

    def test_adapter(self):
        from v2.probe.prompts import s0_messages
        try:
            from v2.harness import prompt_parts
        except ImportError:
            self.skipTest('TODO(WS1): builder unavailable')
        for domain, text in [('scan','walk'),('chainops','Start with [1, 2]. Reverse the list.')]:
            path = ROOT/'v2/domains/prompts'/f'{domain}_v0.json'
            if not path.exists():
                self.skipTest('TODO(WS1): domain prompts unavailable')
            item = dict(domain=domain,input=text)
            self.assertEqual(s0_messages(item),prompt_parts(json.loads(path.read_text()),'S0','direct',input=text))


@unittest.skipUnless(TORCH,'torch unavailable in this environment')
class ExtractionTests(unittest.TestCase):
    def setUp(self):
        from v2.probe.extract import extract_features, write_job
        self.extract = extract_features
        self.write = write_job
        torch.manual_seed(437)
        self.model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,
            intermediate_size=32,num_hidden_layers=3,num_attention_heads=2,
            num_key_value_heads=2,max_position_embeddings=512,attention_dropout=0.0)).eval()
        self.tokenizer = Tokenizer()
        self.texts = ['abcdefgh','ab','abcde','a']
        self.layers = [0,1,3]

    def test_order_and_batch_padding(self):
        for side in ['left','right']:
            self.tokenizer.padding_side = side
            last,mean,n = self.extract(self.model,self.tokenizer,self.texts,self.layers,3)
            one_l,one_m,one_n = self.extract(self.model,self.tokenizer,self.texts,self.layers,1)
            np.testing.assert_array_equal(n,[8,2,5,1])
            np.testing.assert_array_equal(n,one_n)
            np.testing.assert_allclose(last,one_l,atol=2e-6,rtol=2e-5)
            np.testing.assert_allclose(mean,one_m,atol=2e-6,rtol=2e-5)

    def test_pooling_reference(self):
        last,mean,_ = self.extract(self.model,self.tokenizer,self.texts,self.layers,3)
        with torch.inference_mode():
            for i,text in enumerate(self.texts):
                ids = torch.tensor([self.tokenizer(text)['input_ids']])
                h = self.model(input_ids=ids,output_hidden_states=True).hidden_states
                for j,layer in enumerate(self.layers):
                    np.testing.assert_allclose(last[j,i],h[layer][0,-1].numpy(),atol=2e-6)
                    np.testing.assert_allclose(mean[j,i],h[layer][0].mean(0).numpy(),atol=2e-6)

    def test_files_atomic_and_overwrite(self):
        items = [dict(id=str(i),domain='scan',split='router_train',input=t) for i,t in enumerate(self.texts)]
        builder = lambda i: ('s',i['input'])
        with tempfile.TemporaryDirectory(dir=ROOT/'results/v2/features') as folder:
            dest = Path(folder)/'features'
            index = self.write(self.model,self.tokenizer,items,builder,dest,'qwen3:8b',layers=self.layers,quant='float32')
            readback = json.loads((dest/'index.json').read_text())
            for key in ['ids','splits','n_tokens','domain','ollama_model','hf_model_id','revision','quantization','dtype','saved_dtype','layers','poolings','files','hidden_size','n_layers','prompt_template','items_files','code_sha256','batch_size','wall_seconds','load_seconds','total_prompt_tokens','items_per_second','tokens_per_second','cuda_max_memory_allocated','torch_version','transformers_version','timestamp']:
                self.assertIn(key,readback)
            self.assertEqual(readback['ids'],['0','1','2','3'])
            last,mean,_ = self.extract(self.model,self.tokenizer,[format_prompt('qwen3:8b',*builder(i)) for i in items],self.layers,4)
            for j,layer in enumerate(self.layers):
                for pool,expected in [('last',last),('mean',mean)]:
                    name = f'L{layer:02d}_{pool}.npy'
                    array = np.load(dest/name)
                    self.assertEqual(array.shape,(4,16))
                    self.assertEqual(array.dtype,np.float16)
                    np.testing.assert_array_equal(array,expected[j].astype(np.float16))
            with self.assertRaises(FileExistsError):
                self.write(self.model,self.tokenizer,items,builder,dest,'qwen3:8b')
            bad = Path(folder)/'failed'
            with patch('numpy.save',side_effect=OSError('simulated disk failure')):
                with self.assertRaises(OSError):
                    self.write(self.model,self.tokenizer,items,builder,bad,'qwen3:8b')
            self.assertFalse(bad.exists())
            self.assertEqual(list(Path(folder).iterdir()),[dest])
            items[0]['split']='ood_length'
            with self.assertRaises(ValueError):
                self.write(self.model,self.tokenizer,items,builder,bad,'qwen3:8b')
            self.assertFalse(bad.exists())


if __name__=='__main__':
    unittest.main()
