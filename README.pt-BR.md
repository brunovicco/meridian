# Meridian

> 🇺🇸 [Read in English](./README.md)

Uma implementação de referência de uma **plataforma de conhecimento de engenharia** interna: o desenvolvedor faz uma pergunta em linguagem natural, o sistema roteia a consulta, recupera o conhecimento que o desenvolvedor tem permissão de ver e retorna uma resposta fundamentada com citações.

O projeto foi criado para demonstrar um conjunto específico de práticas de engenharia de produção de ponta a ponta (roteamento semântico, contrato de saída de LLM, RAG com controle de acesso e separação limpa de responsabilidades) sobre uma stack de Python, Redis Stack e provedores de embedding/LLM plugáveis.

> **Funciona sem nenhuma configuração.** A configuração padrão usa um embedder fake determinístico, um vector store em memória e um LLM fake baseado em regras, para que você veja o sistema inteiro funcionando sem API key, sem rede e sem Docker:
>
> ```bash
> uv sync
> uv run python -m meridian.interfaces.cli.main --demo
> ```

---

## O que demonstra

| Conceito | Onde | O que observar |
|---|---|---|
| **Roteador semântico** | `application/router/` | Scoring com penalidade negativa, três regras de ambiguidade, matrizes cacheadas por fingerprint |
| **Motor de roteamento** | `application/router/routing_engine.py` | Sinal → decisão, separado da matemática pura |
| **Contrato de saída de LLM** | `application/services/query_understanding.py` | Schema Pydantic + coerção que absorve desvios de formato (estilo DSPy) |
| **RAG com controle de acesso** | `application/pipelines/rag_pipeline.py` | Filtro ACL em tempo de recuperação, citações obrigatórias, "não sei" honesto |
| **Modelo fat/slim** | `domain/models/knowledge.py`, stores | Projeção slim para busca, documento fat buscado sob demanda via JSON.GET |
| **DSPy (real) + Groq** | `infrastructure/dspy/` | `dspy.Predict` para roteamento + `dspy.Refine` com recompensa de fundamentação, no Groq; fallback fake por padrão |
| **Clean Architecture/SOLID** | toda a árvore | Dependências apontam para dentro; concretizações escolhidas apenas na composition root |
| **Configuração twelve-factor** | `infrastructure/config/settings.py` | Todas as configurações via variáveis de ambiente |
| **Vector store Redis Stack** | `infrastructure/redis/` | KNN com RediSearch e filtro ACL via metadados |
| **Consulta estruturada (RediSearch)** | `application/query/` | Filtro tipado → `FT.SEARCH` compilado sobre o catálogo de serviços, escopo ACL, sanitizado contra injeção |
| **Métricas de roteamento** | `infrastructure/metrics/` | Contadores Redis HASH (`HINCRBY` + TTL rolante) com detecção de degradação por taxa de fallback |
| **Stack de entidades (anáfora)** | `application/services/entity_stack.py` | LIFO limitado de entidades discutidas, serializável em JSON para cache Redis com TTL |
| **Observabilidade** | `infrastructure/observability/` | Evento estruturado por decisão de roteamento e recuperação |

---

## O fluxo de uma requisição

```
                        ┌──────────────────────────────────────────────┐
   "como configuro      │                 AskService                   │
    autenticação?"  ─►  │  (application/services/ask_service.py)       │
                        └───────────────┬──────────────────────────────┘
                                        │
                        1. SemanticRouter.route()    ── camada 1: scores + ambiguidade
                                        │
                        2. RoutingEngine.decide()    ── camada 2: sinal → ação
                                        │
                        3. QueryUnderstanding        ── LLM atrás de um contrato Pydantic
                           (coerção absorve desvios)
                                        │
                        4. RagPipeline.run()         ── recuperação com filtro ACL,
                                        │                 geração fundamentada, citações
                                        ▼
                                 Answer + Citations
```

Dois pipelines rodam em momentos distintos e o código os mantém separados:

- **Ingestão (offline):** documentos são fragmentados, embedados e indexados com seus metadados ACL. Aqui os dados vêm de `data/catalog/knowledge_base.json`.
- **Consulta (online):** o fluxo acima.

---

## O roteador semântico, concretamente

Cada intenção é definida por frases **positivas** (como ela se parece) e frases **negativas** (confusíveis de outras intenções). Na construção, essas frases são embedadas em matrizes por intenção e cacheadas no vector store sob um **fingerprint SHA-256** do catálogo, dos thresholds e da dimensão do embedding - mude qualquer um desses e o cache invalida automaticamente.

Para um vetor de consulta `q`, cada intenção recebe uma pontuação:

```
score = max(M_pos @ q) − NEG_PENALTY · max(0, max(M_neg @ q))
```

O `max` sobre as linhas recompensa o exemplo mais próximo (não a média), e o termo negativo (limitado em zero, nunca soma) empurra a fronteira para longe dos confusíveis conhecidos. Em seguida, três regras de ambiguidade são aplicadas em ordem:

1. **Threshold por intenção** - pontuação máxima abaixo do threshold da intenção vencedora.
2. **Piso absoluto** (`AMBIG_MIN`) - pontuação muito fraca em geral, a menos que a margem sobre o segundo colocado seja confortável.
3. **Margem** (`AMBIG_DELTA`) - os dois primeiros muito próximos para separar com segurança.

O motor de roteamento transforma esse sinal em uma de três ações: rotear diretamente, pedir desambiguação ou cair no QA genérico.

> A matemática de scoring fica em `application/router/scoring.py`, é completamente pura (sem I/O) e testada com matrizes construídas manualmente.

---

## Controle de acesso é um filtro em tempo de recuperação

A propriedade de segurança mais importante: **um usuário nunca recupera um chunk fora dos seus grupos, nem transitoriamente.** A verificação ACL é um filtro de metadados *dentro* da busca vetorial (uma cláusula de tag RediSearch combinada com a cláusula KNN) não um passo aplicado após a recuperação, e nunca delegado ao LLM.

Veja isoladamente:

```bash
uv run python -m meridian.interfaces.cli.main --acl-demo
```

```
ACL probe: retrieving 'security post mortem for the payments outage root cause'

  [carol groups=security             ] -> Security Post-Mortem, Credential Rotation Guide
  [alice groups=payments,platform    ] -> Payments Service Auth Guide, Database Failover Runbook, ...
  [dan   groups=(no groups)          ] -> (nothing visible)
```

Carol (security) vê o post-mortem restrito; Alice (payments/platform) nunca o vê; Dan (sem grupos) não vê nada - o filtro falha fechado.

---

## Conhecimento estruturado é um problema de consulta, não de recuperação

Nem todo conhecimento é prosa não estruturada. "Quem é dono do serviço de pagamentos" ou "quais serviços tier-1 não têm dono" são perguntas sobre um **catálogo de serviços** - dados estruturados onde a resposta correta é completa, não um top-K amostrado. Jogar linhas do catálogo num contexto de LLM retorna uma fração; compilar a pergunta numa consulta retorna a resposta inteira. A rota `structured_query` faz isso.

```bash
uv run python -m meridian.interfaces.cli.main --structured-demo
```

```
[alice] Q: who owns the payments service
    compiled: @visibility:{payments | platform} @domain:{payments}
      - payments-api (team payments, tier1)
      - transfer-service (team payments, tier1)

[bob]   Q: list tier1 services in the gateway domain
    compiled: @visibility:{sre | platform} @domain:{gateway} @tier:{tier1}
      - api-gateway (team platform, tier1)
```

O `ServiceQueryBuilder` (`application/query/`) classifica cada campo de filtro como **TAG** (exato), **TEXT** (fuzzy com regras de tokenização) ou **NUMERIC** (intervalo) e compila uma expressão RediSearch. Duas propriedades são estruturais: a cláusula de visibilidade é sempre prefixada (nenhum chamador consegue construir uma consulta sem escopo) e o resultado passa por um **sanitizador** que rejeita verbos de agregação, entrada excessivamente longa e caracteres de controle, falhando de forma segura para um wildcard.

---

## Um documento, duas representações: fat/slim

O mesmo documento de conhecimento vive no Redis Stack como dois payloads, cada um dimensionado para um estágio diferente. A busca paga pelo menor; o maior é buscado apenas para os poucos documentos que sobrevivem ao ranking.

```bash
uv run python -m meridian.interfaces.cli.main --fatslim-demo
```

```
fat/slim probe: 'how do I configure authentication...' as alice

  Phase 1 - slim search (cheap, projections only):
    · Payments Service Authentication   [snippet: To configure authentication for the payments...]
    · Database Failover Procedure       [snippet: For a database failover, first confirm...]
    · Gateway Rate Limiting             [snippet: The gateway rate limiter uses a token bucket...]

  Phase 2 - fat fetch (JSON.GET) for survivors only:
    · Payments Service Authentication   owner=payments  updated=2025-11-02  chars=394
    · Database Failover Procedure       owner=sre       updated=2025-12-01  chars=371
```

A **projeção slim** (título, snippet, fonte, ACL) é um hash indexado que o KNN retorna,  pequeno, rápido, suficiente para ranquear e citar. O **documento fat** (texto completo mais metadados ricos) é um corpo RedisJSON buscado por `JSON.GET`, e apenas para os sobreviventes que entrarão no contexto de geração. O `RagPipeline` executa exatamente esse fluxo: `search_slim` → selecionar sobreviventes → `fetch_fat` - então o payload fat é pago poucas vezes por consulta, nunca uma vez por candidato.

---

## DSPy real no Groq, com fallback fake por padrão

Os contratos de roteamento e geração são suportados por **módulos DSPy reais** quando você opta por eles (`dspy.Predict` para roteamento e `dspy.Refine` para geração) rodando no **Groq** via `GROQ_API_KEY`.

```bash
uv sync --extra groq
export GROQ_API_KEY=gsk_...
MERIDIAN_LLM_BACKEND=groq uv run python -m meridian.interfaces.cli.main --demo
```

O `DSPyRefineModule` é o mais interessante: ele gera uma resposta, pontua com uma **recompensa de fundamentação** (ela cita uma fonte? as afirmações se sobrepõem ao contexto? evita suposições não suportadas?) e regera até um orçamento de tentativas até que a pontuação supere o threshold - o padrão de autocorreção de um advisor de compliance de produção, adaptado para um domínio de conhecimento. A saída ainda passa pelo mesmo contrato de coerção Pydantic que o caminho fake, então o desvio é absorvido de forma idêntica.

Crucialmente, **o provedor fake é o padrão**, e o backend Groq degrada para ele graciosamente quando `dspy` ou a key estão ausentes. O demo zero-setup nunca depende da rede, você ativa o Groq deliberadamente, com a key em mãos.

---

## Observando a saúde do roteador

Cada decisão de roteamento é registrada por um coletor de métricas (`infrastructure/metrics/`). Contadores em processo alimentam uma verificação rápida de degradação - se uma parcela grande demais das decisões cai na rota genérica, o roteador pode estar driftando e precisa de recompilação. Um backend Redis espelha os contadores em um único HASH com `HINCRBY` e TTL rolante de 24 horas, que agrega corretamente entre workers. As métricas nunca quebram o caminho de requisição: falhas no backend são engolidas e a escrita durável fica fora do hot path.

---

## Rodando com infraestrutura real

O ponto central das abstrações é que o mesmo código de aplicação roda contra backends diferentes. Para usar o **Redis Stack** em vez do store em memória:

```bash
uv sync --extra redis
docker compose up -d          # Redis Stack em :6379, RedisInsight em :8001
MERIDIAN_BACKEND=redis uv run python -m meridian.interfaces.cli.main --demo
```

Para usar embeddings/LLM do **Azure OpenAI**, configure `MERIDIAN_EMBEDDING_BACKEND=azure` e `MERIDIAN_LLM_BACKEND=azure` e forneça as variáveis Azure (veja `.env.example`). As classes de provedor em `infrastructure/embeddings/` e `infrastructure/llm/` trazem o scaffolding de produção (retry com backoff e jitter, confiança TLS corporativa) com a chamada SDK marcada como a única lacuna documentada, preenchê-la não toca nenhuma outra camada.

Para usar um embedder semântico **real e gratuito**, sem credencial nenhuma, configure `MERIDIAN_EMBEDDING_BACKEND=local` e `MERIDIAN_EMBEDDING_DIM=384` depois de `uv sync --extra local`. Isso roda o `sentence-transformers/all-MiniLM-L6-v2` localmente (leve, ~80MB, fica em cache após o primeiro download) via `SentenceTransformerEmbeddingProvider` - diferente do skeleton do Azure, esse caminho está totalmente implementado, então é a forma mais rápida de ver roteamento e recuperação semânticos de verdade em vez da aproximação lexical do embedder fake.

Essa substituibilidade é o Princípio de Inversão de Dependência na prática: a troca acontece em `interfaces/composition.py`, um `if` por componente, e nada acima muda.

---

## Estrutura do projeto

```
src/meridian/
  domain/            # modelos (incl. fat/slim), interfaces, políticas; puro, sem I/O
  application/       # roteador, motor, pipelines RAG + estruturado, query builder, módulos dspy
  infrastructure/    # embeddings, vector/catalog stores, redis, llm (fake/azure/groq), métricas, dspy
  interfaces/        # composition root, CLI
data/catalog/        # intenções + base de conhecimento fat + catálogo de serviços (dados versionados)
tests/               # unitários (peças puras) + integração (fluxos completos, incl. ACL, estruturado, fat/slim)
```

---

## Desenvolvimento

```bash
make install     # instalação editável com extras de dev
make demo        # demo end-to-end com script
make acl-demo    # filtro de controle de acesso em isolamento
make structured-demo  # consulta estruturada compilada para RediSearch
make fatslim-demo     # divisão de recuperação fat/slim
make test        # suite de testes (52 testes)
make check       # lint + typecheck + teste
make redis-up    # iniciar Redis Stack
```

Convenções: todo o código em inglês, docstrings em todos os módulos/classes/funções públicas (estilo Google), type hints completos, `ruff` para lint e formatação. O harness de agente em [`CLAUDE.md`](./CLAUDE.md) documenta as regras de arquitetura e os guardrails que não devem regredir.

---

## Notas sobre o escopo

Este é um repositório de estudo/demo. Os provedores fake são léxicos, não semânticos; eles exercitam o encanamento de forma determinística, não a qualidade de um embedder real. Os thresholds do roteador são fornecidos com uma calibração separada para o backend fake (`domain/policies`), o que por si só é um ponto relevante: **thresholds são uma propriedade do modelo de embedding, então trocar de modelo significa recalibrar, não editar prompts.** O caminho DSPy + Groq é real (`dspy.Predict` + `dspy.Refine` com recompensa de fundamentação) e funciona assim que você instala o extra `groq` e define `GROQ_API_KEY`; sem eles o sistema cai no provedor fake para que o demo padrão sempre rode. Os provedores Azure estão scaffolded até o ponto em que a única peça faltando é a chamada SDK externa.
