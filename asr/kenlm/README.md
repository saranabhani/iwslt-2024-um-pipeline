# KenLM

Language model inference code by Kenneth Heafield (kenlm at kheafield.com)
Full documentation and instructions can be found [here](https://kheafield.com/code/kenlm/).

## Compiling

In summary the following commands will install the KenLM package and create a separate build folder for the cmake process.

<pre>
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
</pre>

## Training KenLM

The following command trains a KenLM according to the text file provided. Substitute 'path_to_text_file' with your text file, and substitute '3' with the number of n-grams you require.

<pre>
cat path_to_text_file | ./bin/lmplz -o 3 > kenlm.arpa
</pre>

## Converting Arpa to Binary file

A binary format makes the file run faster.
<pre>
./bin/build_binary -s kenlm.arpa kenlm.binary
</pre>

## Scorer Scripts
Follow instructions [here](https://deepspeech.readthedocs.io/en/v0.8.2/Scorer.html).

