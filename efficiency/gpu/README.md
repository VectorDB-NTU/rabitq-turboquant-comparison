# GPU Efficiency

Benchmarks TurboQuant and RaBitQ quantization time on GPU across three datasets (GloVe-200d, OpenAI-1536d, OpenAI-3072d).

> **Prerequisite:** Download the datasets first — see the parent [README.md](../README.md).

## TurboQuant

Measure quantization time (4-bit) for all three dimensions:

```bash
for dim in 200 1536 3072; do
    python turbo_quant.py --dim $dim --bitwidth 4 --metric time
done
```

## RaBitQ

First, follow the guide in [./RaBitQ-GPU/README.md](./RaBitQ-GPU/README.md) to build `test_quantizer_standalone` executable, then go to the folder (`./RaBitQ-GPU/bin`) and execute the binary executable:

```bash
./test_quantizer_standalone /path/to/train_dataset 4 false 0 /path/to/test_dataset 0 matrix
```

To reproduce the results in the paper, simply replace `/path/to/train_dataset` and `/path/to/test_dataset` with the paths to the datasets prepared before (GloVe-200d, OpenAI-1536d, OpenAI-3072d). For preparing `.fvecs`, please refer to the [README.md for efficiency evaluation of RaBitQ-CPU](../cpu/README.md). If you have already gotten them prepared, you can execute the following commands for evaluation:

```bash
./test_quantizer_standalone ../../../cpu/glove200_train.fvecs 4 false 0 - 0 matrix
./test_quantizer_standalone ../../../cpu/dbpedia-openai3-1536-train.fvecs 4 false 0 - 0 matrix
./test_quantizer_standalone ../../../cpu/dbpedia-openai3-3072-train.fvecs 4 false 0 - 0 matrix
./test_quantizer_standalone ../../../cpu/glove200_train.fvecs 4 true 0 - 0 fhtkac
./test_quantizer_standalone ../../../cpu/dbpedia-openai3-1536-train.fvecs 4 true 0 - 0 fhtkac
./test_quantizer_standalone ../../../cpu/dbpedia-openai3-3072-train.fvecs 4 true 0 - 0 fhtkac
```
