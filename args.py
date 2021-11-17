# parameters configuration
# for a complete list here:
# https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments


class Args:
    def __init__(
        self,
        output_dir="./models/fine-tuned/dialogpt/small",
        model_type="gpt2",
        model_name_or_path="microsoft/DialoGPT-small",
        config_name="microsoft/DialoGPT-small",
        tokenizer_name="microsoft/DialoGPT-small",
        cache_dir="./models/pre-trained/dialogpt/small",
        do_train=True,
        do_eval=True,
        evaluate_during_training=False,
        per_gpu_train_batch_size=4,
        per_gpu_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=2,
        max_steps=-1,
        warmup_steps=0,
        logging_steps=-1,
        save_steps=3500,
        save_total_limit=None,
        eval_all_checkpoints=False,
        no_cuda=False,
        overwrite_output_dir=True,
        overwrite_cache=True,
        should_continue=False,
        fp16=False,
        fp16_opt_level="O1",
        wandb_project="steve-chatbot",
        # use_multiprocessing= False
    ):

        self.output_dir = output_dir
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.config_name = config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        self.do_train = do_train
        self.do_eval = do_eval
        self.evaluate_during_training = evaluate_during_training
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.eval_all_checkpoints = eval_all_checkpoints
        self.no_cuda = no_cuda
        self.overwrite_output_dir = overwrite_output_dir
        self.overwrite_cache = overwrite_cache
        self.should_continue = should_continue
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.block_size = 512  # input sequence length after tokenization
        self.seed = 42
        self.local_rank = -1
        self.wandb_project = wandb_project
