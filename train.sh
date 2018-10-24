
export CUDA_VISIBLE_DEVICES=3
#python run.py --worker_gpu=2 --hparams='batch_size=4096' --data_dir=/data/lz/t2t/MY-T2T/data --problems=translation --model=transformer --hparams_set=transformer_base_single_gpu --train_steps=200000 --eval_steps=0  --keep_checkpoint_max=2 --output_dir=/data/lz/t2t/MY-T2T/train

#nohup python run.py --hparams='batch_size=4096,shared_embedding_and_softmax_weights_source=0' --data_dir=/data/lz/t2t/MY-T2T/data --vocab_size=30720 --vocab_src_name=vocab.bpe.zh --vocab_tgt_name=vocab.bpe.en --problems=translation --model=transformer --train_steps=200000 --eval_steps=0  --keep_checkpoint_max=2 --output_dir=/data/lz/t2t/MY-T2T/train-v18-st > log.train.v18-st &

nohup python run.py --hparams='batch_size=4096,shared_embedding_and_softmax_weights_source=1' --data_dir=/data/lz/t2t/MY-T2T/data --vocab_size=30720 --vocab_src_name=vocab.bpe.zh --vocab_tgt_name=vocab.bpe.en --problems=translation --model=transformer --train_steps=200000 --eval_steps=0  --keep_checkpoint_max=2 --output_dir=/data/lz/t2t/MY-T2T/train-v18-base > log.train.v18-base &
