

python run.py \
		--generate_data=True \
		--data_dir=./data \
		--tmp_dir=./data-tmp \
		--train_src_name=2m.bpe.zh \
		--train_tgt_name=2m.bpe.en \
		--vocab_src_size=30720 \
		--vocab_tgt_size=30720 \
		--vocab_src_name=vocab.zh \
		--vocab_tgt_name=vocab.en \
		--num_shards=25 


