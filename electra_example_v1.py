# reference: https://github.com/retarfi/language-pretraining

# config
"electra-base" : {
    "number-of-layers" : 12,
    "hidden-size" : 768,
    "sequence-length" : 512,
    "ffn-inner-hidden-size" : 3072,
    "attention-heads" : 12,
    "embedding-size" : 768,
    "generator-size" : "1/3",
    "mask-percent" : 15,
    "warmup-steps" : 10000,
    "learning-rate" : 2e-4,
    "batch-size" : {
        "-1" : 256
    },
    "train-steps" : 766000
},


