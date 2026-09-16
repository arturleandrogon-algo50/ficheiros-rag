# Fichário — Assistente de Documentos com RAG

Aplicação web que implementa RAG (*Retrieval-Augmented Generation*) inteiramente no navegador: você envia um PDF, o app extrai e indexa o conteúdo, e depois responde perguntas com base **apenas** no que está no documento — citando a página e o trecho de onde tirou cada resposta.

Sem backend, sem banco de dados vetorial externo. Só HTML, CSS e JavaScript puro, consumindo a API do Google Gemini.

## Como funciona (pipeline RAG)

1. **Extração** — o PDF.js lê o texto de cada página do PDF diretamente no navegador.
2. **Chunking** — o texto é dividido em pedaços menores ("fichas"), com sobreposição entre eles para não cortar frases importantes ao meio.
3. **Embeddings** — cada ficha é transformada em um vetor numérico via API do Gemini (`gemini-embedding-001`), que representa seu significado semântico.
4. **Busca por similaridade** — quando você faz uma pergunta, ela também vira um vetor, e o app compara (similaridade de cosseno) com todas as fichas para achar as mais relevantes.
5. **Geração da resposta** — as fichas mais relevantes são enviadas como contexto para o modelo de linguagem (`gemini-3.6-flash`), que responde só com base nelas — e não inventa informação que não está no documento.

## Como usar

1. Abra o `index.html` no navegador (Chrome ou Edge).
2. Clique em **Chave de API** e cole uma chave gratuita do Google Gemini ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)). A chave fica salva só no seu navegador (`localStorage`), nunca é enviada a nenhum servidor além da própria API do Google.
3. Arraste um PDF para a gaveta à esquerda e aguarde o processamento (extração → fichas → embeddings).
4. Faça perguntas sobre o conteúdo do documento no chat. Cada resposta vem acompanhada das fichas usadas como fonte, com a página e o percentual de similaridade.

## Tecnologias

- HTML, CSS e JavaScript puro (sem frameworks)
- [PDF.js](https://mozilla.github.io/pdf.js/) para extração de texto
- API do Google Gemini para embeddings e geração de texto
- Busca por similaridade de cosseno implementada do zero



## Conceito visual

A interface usa a metáfora de um fichário/arquivo de biblioteca: cada trecho do documento processado vira uma "ficha catalográfica" visível na lateral. A ideia é deixar visível uma etapa que normalmente é invisível em apps de IA — a etapa de *retrieval* (busca) que fundamenta a resposta do modelo.

## 🔬 Modo Fine-tuning (RAG vs. Fine-tuning)

Como extensão deste projeto, foi adicionado um segundo modo de resposta, usando um modelo ajustado via **fine-tuning (LoRA)** sobre a mesma base de conhecimento do modo RAG — permitindo comparar as duas abordagens lado a lado.

### Por que comparar as duas abordagens?

| | RAG (modo padrão) | Fine-tuning (novo modo) |
|---|---|---|
| Como usa o conhecimento | Busca o trecho relevante em tempo real | O conhecimento é incorporado nos parâmetros do modelo durante o treino |
| Cita a fonte | ✅ Sim | ❌ Não |
| Atualização de conteúdo | Só adicionar novos documentos | Precisa re-treinar o modelo |
| Custo/complexidade | Menor | Maior (treino, GPU, dataset) |

### Como o fine-tuning foi feito

1. Geração de um dataset de perguntas e respostas (`.jsonl`) a partir dos mesmos documentos usados no RAG
2. Fine-tuning via **LoRA** (Parameter-Efficient Fine-Tuning), usando o modelo `Phi-3-mini` no Google Colab (GPU gratuita)
3. Publicação do modelo ajustado no Hugging Face Hub
4. Integração do modelo ao Fichário como um segundo modo de resposta

O notebook completo do treino está disponível em [`/fine_tuning/fine_tuning_fichario.ipynb`](./fine_tuning/fine_tuning_fichario.ipynb).

### Stack adicional

- Hugging Face (`transformers`, `peft`, `datasets`)
- LoRA (fine-tuning eficiente em parâmetros)
- Google Colab (treino com GPU gratuita)
