# MPT-7B LoRA Patch

This is the Python model code for MPT-7B patched so that it can be used with a LoRA. Note that while I tested that it works and I get reasonable results out, it is very possible that the model isn't being trained correctly. The model code specifically says that left padding is not supported, but I forcibly did so and got decent results.

Note that when using LoRA, there is a strange quirk that prevents me from causing generation with an empty prompt.

If you would like to use this with `text-generation-webui`, apply the following patch:

```patch
diff --git a/modules/training.py b/modules/training.py
index 278291c..d3ad18c 100644
--- a/modules/training.py
+++ b/modules/training.py
@@ -28,12 +28,13 @@ try:
     MODEL_CLASSES = {v: k for k, v in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES}
 except:
     standard_modules = ["q_proj", "v_proj"]
-    model_to_lora_modules = {"llama": standard_modules, "opt": standard_modules, "gptj": standard_modules, "gpt_neox": ["query_key_value"]}
+    model_to_lora_modules = {"llama": standard_modules, "opt": standard_modules, "gptj": standard_modules, "gpt_neox": ["query_key_value"], "mpt": ["Wqkv"]}
     MODEL_CLASSES = {
         "LlamaForCausalLM": "llama",
         "OPTForCausalLM": "opt",
         "GPTJForCausalLM": "gptj",
-        "GPTNeoXForCausalLM": "gpt_neox"
+        "GPTNeoXForCausalLM": "gpt_neox",
+        "MPTForCausalLM": "mpt"
     }

 WANT_INTERRUPT = False
```

You can find the source model here: [mosaicml/mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)

The alterations are based on the [source code for the llama model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) from HF Transformers.

## Model License

CC-By-SA-3.0
