# Fichário — Assistente de Documentos com RAG

Aplicação web que implementa RAG (*Retrieval-Augmented Generation*) inteiramente no navegador: você envia um PDF ou uma imagem (JPG/PNG), o app extrai e indexa o conteúdo, e depois responde perguntas com base **apenas** no que está no documento — citando a página e o trecho de onde tirou cada resposta.

Sem backend, sem banco de dados vetorial externo. Só HTML, CSS e JavaScript puro, consumindo a API do Google Gemini.

## Como funciona (pipeline RAG)

1. **Extração** — o PDF.js lê o texto de cada página do PDF diretamente no navegador. Páginas escaneadas (sem texto) e imagens JPG/PNG são lidas por OCR local com o Tesseract.js.
2. **Chunking** — o texto é dividido em pedaços menores ("fichas"), com sobreposição entre eles para não cortar frases importantes ao meio.
3. **Embeddings** — cada ficha é transformada em um vetor numérico via API do Gemini (`gemini-embedding-001`), que representa seu significado semântico.
4. **Busca por similaridade** — quando você faz uma pergunta, ela também vira um vetor, e o app compara (similaridade de cosseno) com todas as fichas para achar as mais relevantes.
5. **Geração da resposta** — as fichas mais relevantes são enviadas como contexto para o modelo de linguagem (`gemini-3.6-flash`), que responde só com base nelas — e não inventa informação que não está no documento.

## Como usar

1. Abra o `index.html` no navegador (Chrome ou Edge).
2. Clique em **Chave de API** e cole uma chave gratuita do Google Gemini ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)). A chave fica salva só no seu navegador (`localStorage`), nunca é enviada a nenhum servidor além da própria API do Google.
3. Arraste um PDF ou uma imagem (JPG/PNG) para a gaveta à esquerda e aguarde o processamento (extração → fichas → embeddings).
4. Faça perguntas sobre o conteúdo do documento no chat. Cada resposta vem acompanhada das fichas usadas como fonte, com a página e o percentual de similaridade.

## Tecnologias

- HTML, CSS e JavaScript puro (sem frameworks)
- [PDF.js](https://mozilla.github.io/pdf.js/) para extração de texto
- [Tesseract.js](https://tesseract.projectnaptha.com/) para OCR local de PDFs escaneados e imagens (JPG/PNG)
- API do Google Gemini para embeddings e geração de texto
- Busca por similaridade de cosseno implementada do zero

## Limitações

- **O que sai do computador:** o arquivo (PDF ou imagem) nunca é enviado a nenhum servidor — a leitura e o OCR rodam no navegador. Já o **texto extraído, inclusive o obtido por OCR, é enviado à API do Google Gemini** para gerar os embeddings, e os trechos mais relevantes vão junto com cada pergunta para gerar a resposta.
- **OCR local é mais lento:** páginas sem texto (scans) e imagens são lidas com o Tesseract.js, o que leva alguns segundos por página. Páginas que já têm texto continuam sendo lidas direto pelo PDF.js, então PDFs mistos funcionam.
- **Primeiro uso do OCR:** o Tesseract.js baixa da CDN o motor de OCR e os dados do idioma português (alguns MB). Depois disso fica em cache no navegador.
- **Qualidade do OCR:** depende da imagem — baixa resolução, páginas tortas, manuscritos, tabelas e colunas podem gerar erros de leitura. Por isso, as fichas vindas de OCR levam a etiqueta **OCR** no índice e nas fontes, e o modelo é avisado de que esses trechos podem conter erros.
- **Idioma:** o OCR está configurado só para português.
- **Imagens:** aceita apenas JPG e PNG, uma imagem por vez.

## Conceito visual

A interface usa a metáfora de um fichário/arquivo de biblioteca: cada trecho do documento processado vira uma "ficha catalográfica" visível na lateral. A ideia é deixar visível uma etapa que normalmente é invisível em apps de IA — a etapa de *retrieval* (busca) que fundamenta a resposta do modelo.

## 🔬 Experimento: RAG vs. Fine-tuning (em preparação)

Além do app, este repositório tem um notebook preparado para comparar RAG com fine-tuning (LoRA). **O Fichário funciona só com RAG. O notebook ainda não foi executado**, então ainda não há modelo treinado nem resultados.

### Por que comparar as duas abordagens?

| | RAG (o Fichário) | Fine-tuning (experimento) |
|---|---|---|
| Como usa o conhecimento | Busca o trecho relevante em tempo real | O conhecimento é incorporado nos parâmetros do modelo durante o treino |
| Cita a fonte | ✅ Sim | ❌ Não |
| Atualização de conteúdo | Só adicionar novos documentos | Precisa re-treinar o modelo |
| Custo/complexidade | Menor | Maior (treino, GPU, dataset) |

### O que o notebook faz

O notebook [`fine_tuning/fine_tuning_fichario.ipynb`](./fine_tuning/fine_tuning_fichario.ipynb) foi feito para rodar no Google Colab com GPU (T4):

1. Instala as dependências (`transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes`)
2. Carrega o modelo base `microsoft/Phi-3-mini-4k-instruct` quantizado em 4-bit, para caber na GPU gratuita
3. Carrega um dataset de perguntas e respostas (`dataset_finetuning.jsonl`)
4. Configura o LoRA: só pequenas camadas extras são treinadas e o modelo original fica congelado
5. Treina por 3 épocas com o `SFTTrainer`
6. Testa o modelo ajustado com uma pergunta
7. Salva os adaptadores LoRA localmente (a publicação no Hugging Face Hub está prevista, mas comentada)

### Próximos passos

- Gerar o `dataset_finetuning.jsonl` a partir dos documentos usados no RAG (o arquivo ainda não está no repositório)
- Rodar o treino no Colab e registrar os resultados aqui
- Comparar as respostas do modelo ajustado com as do Fichário

### Stack do experimento

- Hugging Face (`transformers`, `peft`, `datasets`, `trl`)
- LoRA (fine-tuning eficiente em parâmetros)
- Google Colab (GPU gratuita)
