# 📂 Diretório `data/`

Esta pasta armazena **todas as versões dos dados** utilizados no projeto — desde a coleta inicial até o dataset final tratado e pronto para análise.  
Ela segue uma estrutura padronizada inspirada em boas práticas de *data engineering* e *data science pipelines*.

---

## 🧾 Estrutura

```
data/
├── raw/         # dados brutos, originais
├── interim/     # dados intermediários (pós-limpeza)
└── processed/   # dados finais tratados
```

---

## 📘 Descrição dos Subdiretórios

### 🧱 `raw/` — Dados Brutos  
Contém os **arquivos originais** obtidos de fontes externas (APIs, CSVs, bancos, etc.).  
Esses dados **não devem ser modificados manualmente** — servem como referência imutável da origem.

> Exemplo: `raw/dataset.csv`  
> O arquivo principal de entrada configurado na etapa **SOURCES** do notebook.

---

### ⚙️ `interim/` — Dados Intermediários  
Armazena versões **parciais e limpas** dos datasets após etapas de tratamento:  
remoção de nulos, normalização de texto, detecção de outliers, etc.  

Serve como uma camada **de depuração e checkpoint**, permitindo pausar ou revisar o pipeline sem refazer todas as etapas.

> Exemplo: `interim/dataset_interim.csv`  
> Gerado automaticamente se `export_interim = true` no arquivo `config/defaults.json`.

---

### 📊 `processed/` — Dados Processados  
Reúne os datasets **completamente tratados e prontos para análise ou modelagem**.  
É essa versão que deve ser usada em painéis Power BI, notebooks de ML ou integrações com bancos de dados.

> Exemplo: `processed/dataset_processed.csv`  
> Exportado automaticamente quando `export_processed = true`.

---

## 💡 Boas Práticas

- Nunca edite manualmente os arquivos de `raw/`.  
- Use `interim/` para inspeções e testes durante o desenvolvimento.  
- Considere manter apenas `processed/` em ambientes de produção.  
- Todos os diretórios são criados automaticamente pelo notebook, se não existirem.
