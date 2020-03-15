import sentencepiece as spm

class SPT:
    MODEL_PREFIX = "tokenizer" #@param {type: "string"}
    VOC_SIZE = 32000 #@param {type:"integer"}
    SUBSAMPLE_SIZE = 12800000 #@param {type:"integer"}
    NUM_PLACEHOLDERS = 256 #@param {type:"integer"}

    def __init__(self, PRC_DATA_FPATH):
        self.SPM_COMMAND = ('--input={} --model_prefix={} '
                '--vocab_size={} --input_sentence_size={} '
                '--shuffle_input_sentence=true ' 
                '--bos_id=-1 --eos_id=-1').format(
                PRC_DATA_FPATH, self.MODEL_PREFIX, 
                self.VOC_SIZE - self.NUM_PLACEHOLDERS, self.SUBSAMPLE_SIZE)

    def train_sentencetrainer(self):
        spm.SentencePieceTrainer.Train(self.SPM_COMMAND)