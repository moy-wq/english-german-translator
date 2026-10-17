import logging
from pathlib import Path
from typing import Iterator, List, Dict

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

class BPETokenizerBuilder:
    """

    Constrói e gerencia os Tokenizadores BPE para modelo Encoder-Decoder
    
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.especial_tokens: Dict[str, int] = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
        }
        self.especial_tokens_list: List[str] = list(self.especial_tokens.keyes())

    def _get_corpus_iterator(self, dataset, lang: str, batch_size: int = 10000) -> Iterator:    
        """ Data loader próprio do tokenizador """
        for i in range(0, len(dataset)): 
            yield dataset[i: i + batch_size][lang]

    def build_tokenizer(self, corpus_iterator: Iterator[List[str]])-> Tokenizer:
        """ Construção do Tokenizador """
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

        # separação por espaço antes da tokenização
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() 
        
        trainer = trainers.BpeTrainer(
            vocab_size = self.vocab_size,
            special_tokens = self.especial_tokens_list
        )

        tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
        
        bos_id = tokenizer.token_to_id("[BOS]")
        eos_id = tokenizer.token_to_id("[EOS]")

        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens = [
                ("BOS", bos_id),
                ("EOS", eos_id)
            ]
        ) # injeção silenciosa dos tokens de início e fim de sentenças

        return tokenizer

    def train_and_save(self, hf_dataset_name: str, save_dir: str):
        """ Orquestrador geral de dowload, treinamento e salvamento dos dicionários BPE"""

        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Baixando dataset '{hf_dataset_name}'")
        dataset = load_dataset(hf_dataset_name)
        train_data  = dataset['train']

        for lang in ['en', 'de']:
            logger.info(f"Iniciando treinamento do tokenizador para o idioma: '{lang}'")
            iterator = self._get_corpus_iterator(train_data, lang=lang)

            file_path = output_path / f"tokenizer_{lang}.json"
            
            tokenizer = self.build_tokenizer(iterator)
            tokenizer.save(str(file_path))
            
            logger.info(f"Tokenizador '{lang}' salvo com sucesso em {file_path} ")



if __name__ == "__main__":
    builder = BPETokenizerBuilder()
    builder.train_and_save(hf_dataset_name="bentrevett/multi30k", save_dir="data")

