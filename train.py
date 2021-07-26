import torch
import spacy
import torch.optim as optim
import math
import time
import random
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

from model import *

def tokenize(spacy_lang, text):
    return [token.text for token in spacy_lang.tokenizer(text)]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')

    tokenized = spacy_en.tokenizer('I am a graduate student.')
    for i, token in enumerate(tokenized):
        print(f'Index {i}: {token.text}')

    # ---
    print('---')
    
    # 전처리 정의
    def tokenize_de(text):
        return [token.text for token in spacy_de.tokenizer(text)]
    def tokenize_en(text):
        return [token.text for token in spacy_en.tokenizer(text)]
    SRC = Field(tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
    TRG = Field(tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

    # Load Multi30k dataset
    train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    print(f'The size of training dataset : {len(train_dataset)}')
    print(f'The size of validation dataset : {len(valid_dataset)}')
    print(f'The size of test dataset : {len(test_dataset)}')

    # Check dataset for one sample
    print(vars(train_dataset.examples[30])['src'])
    print(vars(train_dataset.examples[30])['trg'])

    # Generate a vocab
    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)
    print(f"len(SRC): {len(SRC.vocab)}")
    print(f"len(TRG): {len(TRG.vocab)}")

    print('-')
    print(TRG.vocab.stoi["abcabc"]) # 없는 단어: 0
    print(TRG.vocab.stoi[TRG.pad_token]) # 패딩(padding): 1
    print(TRG.vocab.stoi["<sos>"]) # <sos>: 2
    print(TRG.vocab.stoi["<eos>"]) # <eos>: 3
    print(TRG.vocab.stoi["hello"])
    print(TRG.vocab.stoi["world"])
    print('-')

    # Make batch
    BATCH_SIZE = 128
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, valid_dataset, test_dataset),
        batch_size = BATCH_SIZE,
        device=device
    )
    for i, batch in enumerate(train_iterator):
        src = batch.src
        trg = batch.trg

        print(f'The first batch size : {src.shape}')

        for i in range(src.shape[1]):
            print(f'Index {i}: {src[0][i].item()}')
        break
    print('---')

    # Hyperparameter
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HIDDEN_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    # Define the model
    enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, 100)
    dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, 100)
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX).to(device)

    # Init parameter
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable_parameters')

    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
    model.apply(initialize_weights)

    # Optimizer
    LEARNING_RATE = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # Train
    def train(model, iterator, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0

        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()

            output, _ = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    # 모델 평가(evaluate) 함수
    def evaluate(model, iterator, criterion):
        model.eval() # 평가 모드
        epoch_loss = 0

        with torch.no_grad():
            # 전체 평가 데이터를 확인하며
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg

                # 출력 단어의 마지막 인덱스(<eos>)는 제외
                # 입력을 할 때는 <sos>부터 시작하도록 처리
                output, _ = model(src, trg[:,:-1])

                # output: [배치 크기, trg_len - 1, output_dim]
                # trg: [배치 크기, trg_len]

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                # 출력 단어의 인덱스 0(<sos>)은 제외
                trg = trg[:,1:].contiguous().view(-1)

                # output: [배치 크기 * trg_len - 1, output_dim]
                # trg: [배치 크기 * trg len - 1]

                # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
                loss = criterion(output, trg)

                # 전체 손실 값 계산
                epoch_loss += loss.item()

        return epoch_loss / len(iterator)
    
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time() # 시작 시간 기록

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'transformer_german_to_english.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')