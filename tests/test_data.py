import torch
import pytest

from tokenizers import Tokenizer, pre_tokenizers, models, trainers, processors
from src.data import TranslationDataset, TranslationCollate, get_dataloaders, BatchPair




@pytest.fixture(scope='session')
def dummy_tokenizers():
    def _create_token(texts):
        tok = Tokenizer(model=models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()  # separação por palavra
        
        trainer = trainers.BpeTrainer(
            vocab_size=100,
            special_tokens=[
                 "[PAD]", "[UNK]", "[BOS]", "[EOS]" 
            ]
        )
        
        tok.train_from_iterator(texts, trainer=trainer)

        tok.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]", # $A é a sequencia gerada
            special_tokens=[
                ("[BOS]", tok.token_to_id("[BOS]")),
                ("[EOS]", tok.token_to_id("[EOS]"))
            ]
        )

        return tok

    en_texts = ["Hello world", "A quick brown fox", "Short"]
    de_texts = ["Hallo Welt", "Ein schneller brauner Fuchs", "Kurz"]

    return _create_token(en_texts), _create_token(de_texts)


@pytest.fixture
def dummy_pairs():
    return [
        ("Hello world", "Hallo Welt"),
        ("A quick brown fox jumps", "Ein schneller brauner Fuchs springt"),
        ("Short", "Kurz"),
        ("Another longer sentence to test batch padding properly", "Ein weiterer längerer Satz")
        ]


def test_collate_fn_dynamic_padding_and_masks(dummy_tokenizers, dummy_pairs):
    tok_en,tok_de = dummy_tokenizers
    pad_idx = 0
    dataset = TranslationDataset(dummy_pairs, tokenizer_src=tok_en,
                                  tokenizer_tgt=tok_de, max_len=50)

    collate = TranslationCollate(pad_idx)

    batch_raw = [dataset[i] for i in range(len(dataset))]
    batch: BatchPair = collate(batch_raw)

    assert isinstance(batch, BatchPair)
    assert batch.src_token_ids.shape[0] ==  4
    assert batch.tgt_token_ids.shape[0] ==  4

    # Verificação se há padding dinâmico
    batch_size, max_src_len = batch.src_token_ids.shape
    assert batch.src_mask.shape == (batch_size, 1, 1, max_src_len)

    # Verificação se máscara é falsa onde o token é [PAD]
    expected_mask = (batch.src_token_ids != pad_idx).unsqueeze(1).unsqueeze(2)
    assert torch.equal(batch.src_mask, expected_mask)


def test_get_dataloaders_integration(dummy_pairs, dummy_tokenizers):
    """Testa o gerador de dataloaders e o método de transferência .to(device)."""
    tok_en, tok_de = dummy_tokenizers
    train_loader, val_loader, test_val_loader = get_dataloaders(
        train_pairs= dummy_pairs[:2],
        test_pairs= [dummy_pairs[2:3]],
        val_pairs= [dummy_pairs[3:3]], 
        tokenizer_src=tok_en,
        tokenizer_tgt=tok_de,
        batch_size=2,
        max_len=30,
        num_workers=0  # 0 facilita debugging local sem overhead de subprocessos
        )

    # Itera sobre um batch do loader
    for batch in train_loader:
        assert isinstance(batch, BatchPair)
        assert batch.src_token_ids.shape[0] <= 2

        # Testa a movimentação de memória (.to)
        device = torch.device("cpu")
        batch_moved = batch.to(device)
        assert batch_moved.src_token_ids.device == device
        break