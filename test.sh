
export CUDA_VISIBLE_DEVICES=3

python run.py \
		--gpu_mem_fraction=0.6 \
		--hparams='' \
		--data_dir=../data-v21 \
		--hparams_set=transformer_params_base \
		--output_dir=../train-v21-base \
		--vocab_src_size=30720  \
		--vocab_tgt_size=30720  \
		--vocab_src_name=vocab.zh \
		--vocab_tgt_name=vocab.en \
		--train_steps=0 \
		--decode_beam_size=4 \
		--decode_alpha=0.6 \
		--decode_batch_size=50  \
		--decode_from_file=../data/03.seg.bpe \
		--decode_to_file=../output/dev.v21-base.200K.tmp.out 
