#!/bin/bash

WORK=.
DATA_ROOT=$WORK/data/multidomain
DATA_RAW=$DATA_ROOT/data-clean-split
DATA_BIN=$DATA_ROOT/data-bin
DATA_BPE=$DATA_ROOT/data-bpe
BPE_CODE=$DATA_ROOT/joint_bpe.fr-en.32k
l1=en
l2=fr

dict=$DATA_ROOT/dict.all.txt

# preprocessed file
PACKAGES=../../
SCRIPTS=$PACKAGES/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=$PACKAGES/subword-nmt/subword_nmt

mkdir -p $DATA_BIN
mkdir -p $DATA_BPE

####### BPE ALL
# for split in train valid
# do
#     if [[ ! -f $DATA_RAW/multidomain.$split.$l1 ]];
#     then
#         # concatenate all domains for train
#         for lang in $l1 $l2 ${l2}1 ${l2}2 ${l2}3
#         do
#             cat $DATA_RAW/*.$split*.$lang > $DATA_RAW/multidomain.$split.$l1-$l2.$lang
#         done;
#     fi;
    
#     cat $DATA_RAW/multidomain.$split.$l1-$l2.$l1 | \
#     perl $NORM_PUNC $l1 | \
#     perl $REM_NON_PRINT_CHAR | \
#     perl $TOKENIZER -threads 10 -a -l $l1 | \
#     python $BPEROOT/apply_bpe.py -c $BPE_CODE > $DATA_BPE/multidomain.$split.$l1-$l2.$l1

#     cat $DATA_RAW/multidomain.$split.$l1-$l2.$l2 | \
#     perl $NORM_PUNC $l2 | \
#     perl $REM_NON_PRINT_CHAR | \
#     perl $TOKENIZER -threads 10 -a -l $l2 | \
#     python $BPEROOT/apply_bpe.py -c $BPE_CODE > $DATA_BPE/multidomain.$split.$l1-$l2.$l2
# done;

######### BPE PER DOMAIN
# for name in ECB EMEA Europarl GNOME JRC-Acquis KDE4 News-Commentary PHP TED2013 Ubuntu Wikipedia
# do
#     for split in train valid test test-s
#     do
#         cat $DATA_RAW/$name.$split.$l1 | \
#         perl $NORM_PUNC $l1 | \
#         perl $REM_NON_PRINT_CHAR | \
#         perl $TOKENIZER -threads 10 -a -l $l1 | \
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE > $DATA_BPE/$name.$split.$l1

#         cat $DATA_RAW/$name.$split.$l2 | \
#         perl $NORM_PUNC $l2 | \
#         perl $REM_NON_PRINT_CHAR | \
#         perl $TOKENIZER -threads 10 -a -l $l2 | \
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE > $DATA_BPE/$name.$split.$l2
#     done;
# done;

###### Fairseq PREPROCESS
echo fairseq-preprocess

# if [[ -f $DATA_BIN/dict.$l1.txt ]]; then
#     rm $DATA_BIN/dict.{$l1,$l2}.txt
# fi
# fairseq-preprocess \
# --source-lang $l1 --target-lang $l2 \
# --srcdict $dict \
# --joined-dictionary \
# --trainpref  $DATA_BPE/multidomain.train \
# --validpref $DATA_BPE/multidomain.valid \
# --destdir $DATA_BIN/ \
# --workers 5

for name in ECB EMEA Europarl GNOME JRC-Acquis KDE4 News-Commentary PHP TED2013 Ubuntu Wikipedia
do
    mkdir -p $DATA_BIN/$name
    fairseq-preprocess \
    --source-lang $l1 --target-lang $l2 \
    --srcdict $dict \
    --joined-dictionary \
    --trainpref  $DATA_BPE/$name.train \
    --validpref $DATA_BPE/$name.valid \
    --testpref $DATA_BPE/$name.test \
    --destdir $DATA_BIN/$name \
    --workers 5

    mkdir -p $DATA_BIN/$name/test-s
    fairseq-preprocess \
    --source-lang $l1 --target-lang $l2 \
    --srcdict $dict \
    --joined-dictionary \
    --testpref $DATA_BPE/$name.test-s \
    --destdir $DATA_BIN/$name/test-s \
    --workers 5
done;


