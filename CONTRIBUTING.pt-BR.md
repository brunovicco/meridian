# Contribuindo para o Meridian

> 🇺🇸 [Read in English](./CONTRIBUTING.md)

O Meridian é um artefato de ensino. Contribuições que aprimorem seu valor educacional são bem-vindas: explicações mais claras, código mais idiomático, melhor cobertura de testes e correções quando a implementação se afasta dos princípios que afirma demonstrar.

## Antes de começar

Leia o `CLAUDE.md` - ele codifica as regras de arquitetura e os guardrails que qualquer mudança deve respeitar. Um pull request que os violar não será aceito.

As restrições mais importantes:

1. **Dependências apontam para dentro.** `domain` não importa nada de `application` ou `infrastructure`. `application` importa apenas de `domain`. Concretizações são escolhidas exclusivamente em `interfaces/composition.py`.
2. **Controle de acesso é um filtro em tempo de recuperação.** Nunca introduza um caminho de código que busque e depois filtre; o ACL deve estar dentro da busca.
3. **Citações são obrigatórias em respostas fundamentadas.**
4. **Os provedores fake devem manter `make demo` funcionando sem nenhuma configuração.** Qualquer mudança deve deixar o demo executável sem credenciais e sem Docker.

## Configuração

As dependências são gerenciadas com [uv](https://docs.astral.sh/uv/) e fixadas
com versão exata em `pyproject.toml`, resolvidas em `uv.lock` para um ambiente
reprodutível.

```bash
make install   # uv sync --extra dev
make demo      # verificar que o sistema funciona de ponta a ponta
make test      # 52 testes, todos devem passar
```

Para adicionar uma dependência use `uv add <pacote>` (ou `uv add --optional <extra>
<pacote>` para um extra opcional). Isso atualiza `pyproject.toml` e `uv.lock`
juntos. Não edite os pins de versão manualmente; rode `uv lock` após qualquer
mudança manual em `pyproject.toml`.

## Fazendo mudanças

- Adicione um teste para cada mudança comportamental. Funções puras recebem testes unitários (rápidos, sem I/O). Fluxos de ponta a ponta recebem testes de integração pelo backend em memória.
- Execute `make check` (lint + typecheck + testes) antes de abrir um PR. O CI executa o mesmo gate.
- Mantenha as docstrings no estilo Google. Explique o *porquê*, não apenas o *quê*.
- Type hints são obrigatórios em todas as assinaturas de funções públicas; o `mypy` impõe isso.

## Estilo de commit

Use prefixos de commit convencional: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`. Uma mudança lógica por commit.

## Abrindo um pull request

- Descreva o que mudou e por quê no corpo do PR.
- Referencie o guardrail relevante no `CLAUDE.md` se sua mudança tocar um caminho crítico (controle de acesso, citações, pureza do scoring, configuração).
- Mantenha os PRs pequenos. Um diff focado e revisável é melhor do que um grande.
