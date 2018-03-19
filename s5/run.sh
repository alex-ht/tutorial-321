#!/bin/bash

# Copyright 2017 Beijing Shell Shell Tech. Co. Ltd. (Authors: Hui Bu)
#           2017 Jiayu Du
#           2017 Xingyu Na
#           2017 Bengu Wu
#           2017 Hao Zheng
# Apache 2.0

# 這份是修改自 kaldi/egs/aishell/s5/run.sh

# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
# Caution: some of the graph creation steps use quite a bit of memory, so you
# should run this on a machine that has sufficient memory.
. ./cmd.sh

# === 參數 ===
data=`pwd`/download
data_url=www.openslr.org/resources/33
stage=0
# ============
. ./path.sh        # path.sh 用來設定 PATH 環境變數
. parse_options.sh # parse_options.sh 在 utils目錄底下
                   # 可以在shell直接輸入參數, 例如 ./run.sh --stage -3

if [ $stage -le -3 ]; then # 用 stage 控制起始執行位置
  local/download_and_untar.sh $data $data_url data_aishell || exit 1;
  local/download_and_untar.sh $data $data_url resource_aishell || exit 1;
fi

if [ $stage -le -2 ]; then
  # Lexicon Preparation,
  local/aishell_prepare_dict.sh $data/resource_aishell || exit 1;

  # Data Preparation,
  local/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript || exit 1;

  # Phone Sets, questions, L compilation
  utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
    "<SPOKEN_NOISE>" data/local/lang data/lang || exit 1;
  # 注意,拿掉 --position-dependent-phones false 以後方便取出"詞圖",
  # 即以詞為邊的圖,紀錄所有可能的詞段落資訊.
fi

if [ $stage -le -1 ]; then
  # 需要安裝Kaldi語言模型工具,預設不會安裝
  # 在 kaldi/tools 執行 ./extras/install_kaldi_lm.sh
  # LM training
  local/aishell_train_lms.sh || exit 1;

  # G compilation, check LG composition
  utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1;
fi

# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
if [ $stage -le 0 ]; then
  # Note: 因為訓練時間會太長, 先用指令取出少量資料
  # 尋找工具的原則
  #   1. 和模型訓練的演算法有無關？有關選steps, 無關選utils.  選 utils
  #   2. 看命名選擇可能的指令
  #     a) ./utils/subset_data_dir.sh
  #     b) ./utils/subset_data_dir_tr_cv.sh
  #   3. 直接輸入不帶入任何參數,可以看到說明文字.
  #      或打開 .sh 看裡面的所有變數,因為說明文字不一定完整
  #      可以得知 b) 是用來分訓練集和驗證集, 我們已經有3個資料集,只是要縮減大小,所以選 a)
  # 在這邊我們先取每位語者5句話
  for x in train dev test; do
    utils/subset_data_dir.sh --per-spk data/$x 5 data/${x}_5utt || exit 1;
  done
  mfccdir=mfcc
  for x in train dev test; do
    steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 10 data/${x}_5utt exp/make_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_5utt exp/make_mfcc/$x $mfccdir || exit 1; # 計算mean 和 variance, 資料正規化用
    utils/fix_data_dir.sh data/${x}_5utt || exit 1; # 部份音檔可能會失敗,這指令可以重新整理檔案只保留可用部份.
  done
  exit 0;
fi

if [ $stage -le 1 ]; then
steps/train_mono.sh --cmd "$train_cmd" --nj 10 \
  data/train_5utt data/lang exp/mono || exit 1;

# Monophone decoding
utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/mono/graph data/dev_5utt exp/mono/decode_dev
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/mono/graph data/test_5utt exp/mono/decode_test

# Get alignments from monophone system.
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  data/train_5utt data/lang exp/mono exp/mono_ali || exit 1;

# train tri1 [first triphone pass]
steps/train_deltas.sh --cmd "$train_cmd" \
 2500 20000 data/train_5utt data/lang exp/mono_ali exp/tri1 || exit 1;

# decode tri1
utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/tri1/graph data/dev_5utt exp/tri1/decode_dev
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/tri1/graph data/test_5utt exp/tri1/decode_test

# align tri1
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  data/train_5utt data/lang exp/tri1 exp/tri1_ali || exit 1;

# train tri2 [delta+delta-deltas]
steps/train_deltas.sh --cmd "$train_cmd" \
 2500 20000 data/train_5utt data/lang exp/tri1_ali exp/tri2 || exit 1;

# decode tri2
utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/tri2/graph data/dev_5utt exp/tri2/decode_dev
steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
  exp/tri2/graph data/test_5utt exp/tri2/decode_test

# train and decode tri2b [LDA+MLLT]
steps/align_si.sh --cmd "$train_cmd" --nj 10 \
  data/train_5utt data/lang exp/tri2 exp/tri2_ali || exit 1;

# Train tri3a, which is LDA+MLLT,
steps/train_lda_mllt.sh --cmd "$train_cmd" \
  2500 20000 data/train_5utt data/lang exp/tri2_ali exp/tri3a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri3a exp/tri3a/graph || exit 1;
steps/decode.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
  exp/tri3a/graph data/dev_5utt exp/tri3a/decode_dev
steps/decode.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
  exp/tri3a/graph data/test_5utt exp/tri3a/decode_test

# From now, we start building a more serious system (with SAT), and we'll
# do the alignment with fMLLR.

steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 \
  data/train_5utt data/lang exp/tri3a exp/tri3a_ali || exit 1;

steps/train_sat.sh --cmd "$train_cmd" \
  2500 20000 data/train_5utt data/lang exp/tri3a_ali exp/tri4a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri4a exp/tri4a/graph
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
  exp/tri4a/graph data/dev_5utt exp/tri4a/decode_dev
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
  exp/tri4a/graph data/test_5utt exp/tri4a/decode_test
fi
steps/align_fmllr.sh  --cmd "$train_cmd" --nj 10 \
  data/train_5utt data/lang exp/tri4a exp/tri4a_ali

# Building a larger SAT system.

steps/train_sat.sh --cmd "$train_cmd" \
  3500 100000 data/train_5utt data/lang exp/tri4a_ali exp/tri5a || exit 1;

utils/mkgraph.sh data/lang_test exp/tri5a exp/tri5a/graph || exit 1;
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
   exp/tri5a/graph data/dev_5utt exp/tri5a/decode_dev || exit 1;
steps/decode_fmllr.sh --cmd "$decode_cmd" --nj 10 --config conf/decode.config \
   exp/tri5a/graph data/test_5utt exp/tri5a/decode_test || exit 1;

steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 \
  data/train_5utt data/lang exp/tri5a exp/tri5a_ali || exit 1;

# getting results (see RESULTS file)
for x in exp/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null

exit 0;
# 以下建議先仔細檢查再執行

# nnet3
local/nnet3/run_tdnn.sh

# chain
local/chain/run_tdnn.sh

# getting results (see RESULTS file)
for x in exp/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null

exit 0;
