
A simple Tensorflow implementation of the Transformer
===

This project is a simple implementation of Tensor2tensor (https://github.com/tensorflow/tensor2tensor) for machine translation.

How to use?
---

* Preprosessing. Prepare the parallel data (token, bpe, vocab, and so on), run ./datagen.sh

* Training. Modify the model params (transformer_params_big or transformer_params_base, basic params are set in models/common_hparms.py), and run ./train.sh

* inference.  Run the command to decode: ./test.sh

Contact
---

If you have questions, suggestions and bug reports, please email wszlong@gmail.com.



