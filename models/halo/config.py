'''
    code by Brandon Theodorou
    Original GPT-2 Paper and repository here: https://github.com/openai/gpt-2
    Original GPT-2 Pytorch Model: https://github.com/huggingface/pytorch-pretrained-BERT
    GPT-2 Pytorch Model Derived From: https://github.com/graykode/gpt-2-Pytorch
'''
class HALOConfig(object):
    def __init__(
            self,
            total_vocab_size=14487,
            code_vocab_size=10205,
            lab_vocab_size=237,
            continuous_vocab_size=15,
            label_vocab_size=25,
            special_vocab_size=3,

            categorical_lab_vocab_size=47,
            continuous_lab_vocab_size=190,
            
            phenotype_labels=25, 
            n_positions=150,
            n_ctx=150,
            n_embd=1440,
            n_layer=12,
            n_head=18,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            
            batch_size=56,
            sample_batch_size=128,
            epoch=50,
            lr=1e-4,
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.label_vocab_size = label_vocab_size
        self.lab_vocab_size = lab_vocab_size
        self.categorical_lab_vocab_size = categorical_lab_vocab_size
        self.continuous_lab_vocab_size = continuous_lab_vocab_size
        self.continuous_vocab_size = continuous_vocab_size
        self.special_vocab_size = special_vocab_size
        self.phenotype_labels = phenotype_labels
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.epoch = epoch
        self.lr = lr
